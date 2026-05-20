from __future__ import annotations

import io
import platform
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path

import httpx
from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn

from onda.registry import ModelEntry, ONDA_DIR

PIPER_VERSION = "2023.11.14-2"
PIPER_RELEASES_BASE = (
    f"https://github.com/rhasspy/piper/releases/download/{PIPER_VERSION}"
)


def _piper_asset_name(system: str, machine: str) -> str:
    """Return the piper release tarball filename for the current platform."""
    if system == "Darwin":
        # No native arm64 release in 2023.11.14-2 — x64 runs via Rosetta on Apple Silicon
        return "piper_macos_x64.tar.gz"
    raise RuntimeError(f"Unsupported platform: {system} {machine}")


def get_piper_binary() -> Path:
    """Return path to piper binary, downloading it if needed."""
    bin_dir = ONDA_DIR / "bin"
    bin_path = bin_dir / "piper"
    if bin_path.exists():
        return bin_path

    bin_dir.mkdir(parents=True, exist_ok=True)
    asset = _piper_asset_name(platform.system(), platform.machine())
    url = f"{PIPER_RELEASES_BASE}/{asset}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / asset
        _stream_to_file(url, tar_path, description="Downloading piper binary")

        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(tmpdir, filter="data")

        # Copy entire piper/ bundle (binary + bundled libs + espeak-ng-data) into bin_dir
        # so the binary can find its @rpath dependencies at runtime.
        piper_bundle = Path(tmpdir) / "piper"
        for item in piper_bundle.iterdir():
            dest = bin_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_path


def pull_model(model: ModelEntry) -> None:
    """Download model files and piper binary to ~/.onda/models/<name>/."""
    get_piper_binary()

    dest = ONDA_DIR / "models" / model.name
    dest.mkdir(parents=True, exist_ok=True)

    _stream_to_file(
        model.onnx_url,
        dest / f"{model.name}.onnx",
        description=f"Downloading {model.name}.onnx",
    )
    _stream_to_file(
        model.config_url,
        dest / f"{model.name}.onnx.json",
        description=f"Downloading {model.name}.onnx.json",
    )


def _stream_to_file(url: str, dest: Path, description: str) -> None:
    """Stream a URL to a file with a Rich progress bar."""
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
    ) as progress:
        task = progress.add_task(description, total=None)
        with httpx.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0)) or None
            progress.update(task, total=total)
            with open(dest, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
                    progress.advance(task, len(chunk))

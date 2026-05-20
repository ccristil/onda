import tarfile
import io
import stat
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from onda.registry import ModelEntry
from onda.downloader import get_piper_binary, pull_model, _piper_asset_name


SAMPLE_MODEL = ModelEntry(
    name="en_US-amy-low",
    backend="piper",
    onnx_url="https://example.com/amy.onnx",
    config_url="https://example.com/amy.onnx.json",
    size_mb=5,
    description="Fast.",
)


def _make_fake_tarball(tmp_path: Path) -> bytes:
    """Create an in-memory tarball containing piper/piper binary."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        content = b"#!/bin/sh\necho piper"
        info = tarfile.TarInfo(name="piper/piper")
        info.size = len(content)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def test_piper_asset_name_arm64():
    name = _piper_asset_name("Darwin", "arm64")
    assert "aarch64" in name or "arm64" in name
    assert "apple" in name or "macos" in name


def test_piper_asset_name_x86():
    name = _piper_asset_name("Darwin", "x86_64")
    assert "x86_64" in name or "x64" in name
    assert "apple" in name or "macos" in name


def test_get_piper_binary_downloads_when_missing(tmp_path, mocker):
    mocker.patch("onda.downloader.ONDA_DIR", tmp_path)
    mocker.patch("onda.downloader.platform.system", return_value="Darwin")
    mocker.patch("onda.downloader.platform.machine", return_value="arm64")

    tarball_bytes = _make_fake_tarball(tmp_path)
    mock_response = MagicMock()
    mock_response.iter_bytes.return_value = iter([tarball_bytes])
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    mocker.patch("onda.downloader.httpx.stream", return_value=mock_response)

    result = get_piper_binary()

    assert result.exists()
    assert result.stat().st_mode & stat.S_IXUSR


def test_get_piper_binary_skips_download_when_exists(tmp_path, mocker):
    mocker.patch("onda.downloader.ONDA_DIR", tmp_path)
    bin_path = tmp_path / "bin" / "piper"
    bin_path.parent.mkdir(parents=True)
    bin_path.touch()

    mock_stream = mocker.patch("onda.downloader.httpx.stream")
    result = get_piper_binary()

    mock_stream.assert_not_called()
    assert result == bin_path


def test_pull_model_downloads_files(tmp_path, mocker):
    mocker.patch("onda.downloader.ONDA_DIR", tmp_path)
    mocker.patch("onda.downloader.get_piper_binary", return_value=tmp_path / "piper")

    call_count = 0

    def fake_stream(method, url, **kwargs):
        nonlocal call_count
        call_count += 1
        content = b"fake-content"
        mock_resp = MagicMock()
        mock_resp.iter_bytes.return_value = iter([content])
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    mocker.patch("onda.downloader.httpx.stream", side_effect=fake_stream)

    pull_model(SAMPLE_MODEL)

    model_path = tmp_path / "models" / "en_US-amy-low"
    assert (model_path / "en_US-amy-low.onnx").exists()
    assert (model_path / "en_US-amy-low.onnx.json").exists()
    assert call_count == 2  # one for .onnx, one for .onnx.json

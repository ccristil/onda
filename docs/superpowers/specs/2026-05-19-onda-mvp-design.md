# onda MVP — Design Spec

**Date:** 2026-05-19
**Status:** Approved

## Overview

`onda` is an Ollama-style CLI for local TTS (text-to-speech) models. Pull, manage, and run open-source TTS models from the command line. No cloud. No subscriptions.

## MVP Scope (v0.1)

- **Platform:** Mac only (arm64 + x86_64)
- **Backend:** Piper only (Kokoro/Orpheus stubbed)
- **Models:** 3 curated Piper voices in registry
- **Tests:** Full suite with mocked I/O (no real downloads/audio in tests)
- **Chunking:** Sequential, no progress bar

## CLI Commands

```
onda list                                   # Rich table: Name | Backend | Size | Status
onda pull <model>                           # Download model + piper binary
onda remove <model>                         # Delete ~/.onda/models/<name>/ (with prompt)
onda info <model>                           # Rich panel with model metadata
onda run <model> "text"                     # Speak inline text
onda run <model> --file input.txt           # Speak from file
onda run <model> "text" --out output.wav    # Save to WAV instead of playing
```

## File Architecture

```
onda/
├── pyproject.toml          # entry: onda = onda.main:app
├── models.json             # model registry
├── tests/
│   ├── test_registry.py
│   ├── test_utils.py
│   ├── test_downloader.py
│   ├── test_piper.py
│   └── test_runner.py
└── onda/
    ├── __init__.py
    ├── main.py             # Typer CLI — all commands
    ├── registry.py         # read models.json, check download status
    ├── downloader.py       # pull model files + piper binary (platform-aware)
    ├── runner.py           # dispatch text → correct backend + chunking
    ├── utils.py            # chunk_text(), read_text_file(), play_wav()
    └── backends/
        ├── __init__.py
        ├── piper.py        # subprocess-based piper binary invocation
        ├── kokoro.py       # stub — NotImplementedError
        └── orpheus.py      # stub — NotImplementedError
```

## Module Contracts

### `pyproject.toml`
- Build backend: hatchling (uvx-compatible)
- Core deps: `typer[all]`, `rich`, `httpx`, `sounddevice`, `soundfile`
- Optional groups: `[kokoro]`, `[orpheus]`, `[docs]`
- Dev deps: `pytest`, `pytest-mock`

### `models.json`

Three curated Piper models:

| Name | Size | Quality |
|------|------|---------|
| `en_US-amy-low` | ~5 MB | Fast/tiny |
| `en_US-lessac-medium` | ~63 MB | Balanced (recommended default) |
| `en_US-ryan-high` | ~100 MB | High quality |

Each entry: `name`, `backend`, `onnx_url`, `config_url`, `size_mb`, `description`.
URLs from `https://huggingface.co/rhasspy/piper-voices`.

### `registry.py`

```python
@dataclass
class ModelEntry:
    name: str
    backend: str
    onnx_url: str
    config_url: str
    size_mb: int
    description: str

def load_registry() -> list[ModelEntry]       # reads bundled models.json
def get_model(name: str) -> ModelEntry | None
def is_downloaded(name: str) -> bool          # checks ~/.onda/models/<name>/ has .onnx + .onnx.json
def model_dir(name: str) -> Path              # ~/.onda/models/<name>/
```

### `downloader.py`

```python
PIPER_VERSION = "2023.11.14-2"

def get_piper_binary() -> Path
    # Downloads piper to ~/.onda/bin/piper if missing
    # platform.system() + platform.machine() → selects .tar.gz
    # Mac arm64 → aarch64-apple-darwin, Mac x86 → x86_64-apple-darwin

def pull_model(model: ModelEntry) -> None
    # 1. get_piper_binary()
    # 2. Stream .onnx + .onnx.json → ~/.onda/models/<name>/
    # Rich progress bar per file, httpx streaming
```

### `utils.py`

```python
def chunk_text(text: str, max_chars: int = 500) -> list[str]
    # Accumulate sentences (". ", "! ", "? ") until chunk exceeds max_chars, then flush
    # Hard-split only if single sentence > max_chars

def read_text_file(path: Path) -> str
    # Reads .txt; raises typer.BadParameter for unsupported types

def play_wav(path: Path) -> None
    # soundfile.read() + sounddevice.play() + sounddevice.wait()
```

### `backends/piper.py`

```python
class PiperBackend:
    model_dir: Path

    def speak(self, text: str, output_path: Path | None = None) -> None:
        # subprocess.run(piper_binary, --model ..., --output_file tmp.wav, stdin=text)
        # if output_path: move tmp → output_path
        # else: play_wav(tmp), delete tmp

def load(model_dir: Path) -> PiperBackend
    # validates .onnx + .onnx.json exist
```

### `runner.py`

```python
def run(model_name: str, text: str, output_path: Path | None = None) -> None:
    # 1. get_model() — error if not in registry
    # 2. is_downloaded() — error + hint to `onda pull` if missing
    # 3. load backend
    # 4. chunk_text(text)
    # 5a. With --out: synthesize chunks to tmp WAVs, concatenate numpy arrays, write final WAV
    # 5b. Without --out: backend.speak(chunk) sequentially for each chunk
```

## Testing Strategy

All tests mock I/O — no real network, audio, or subprocess calls:

| File | What it covers |
|------|---------------|
| `test_registry.py` | load/lookup/is_downloaded with tmp models.json |
| `test_utils.py` | chunk_text edge cases, read_text_file error on bad extension |
| `test_downloader.py` | httpx mocked, assert files written, binary marked executable |
| `test_piper.py` | subprocess.run mocked, play_wav mocked, correct args asserted |
| `test_runner.py` | backend.speak mocked, chunk iteration, WAV concatenation |

## Build Order

1. `pyproject.toml` + `models.json`
2. `registry.py` + `tests/test_registry.py`
3. `utils.py` + `tests/test_utils.py`
4. `downloader.py` + `tests/test_downloader.py`
5. `backends/piper.py` + `tests/test_piper.py`
6. `runner.py` + `tests/test_runner.py`
7. `main.py` + manual smoke test

## Verification

```bash
pip install -e ".[dev]"
pytest

onda list
onda pull en_US-amy-low
onda run en_US-amy-low "Hello, this is onda speaking."
onda run en_US-amy-low --file some_text.txt
onda run en_US-amy-low "Save this." --out output.wav
onda info en_US-amy-low
onda remove en_US-amy-low
```

# onda

Ollama-style CLI for local TTS models. Pull, manage, and run open-source TTS models from the command line.

## Vision

Dead-simple local TTS. No cloud, no subscriptions. Like `ollama` but for text-to-speech.

## Stack

- **Language:** Python 3.10+
- **CLI:** Typer + Rich
- **HTTP:** httpx (model downloads)
- **Audio:** sounddevice + soundfile
- **Packaging:** hatchling, uvx-compatible

## Commands

```
onda list                          # show registry + download status
onda pull <model>                  # download a model
onda remove <model>                # delete a model
onda info <model>                  # show model details
onda run <model> "text"            # speak inline text
onda run <model> --file input.txt  # speak from file
onda run <model> --file input.txt --out audio.wav  # save instead of play
```

## File Architecture

```
onda/
├── CLAUDE.md
├── pyproject.toml          # entry point: onda = onda.main:app
├── models.json             # model registry: name, backend, URLs, metadata
└── onda/
    ├── main.py             # typer CLI — all commands live here
    ├── registry.py         # read models.json, check download status
    ├── downloader.py       # pull model files + piper binary (platform-aware)
    ├── runner.py           # dispatch text → correct backend
    ├── utils.py            # chunk_text(), read_text_file(), play_wav()
    └── backends/
        ├── piper.py        # fast, lightweight (recommended default)
        ├── kokoro.py       # high quality, multiple voices
        └── orpheus.py      # expressive, emotion tags (stub — needs snac)
```

## Backends

| Backend | Quality | Size | Notes |
|---------|---------|------|-------|
| Piper | Good | 5–63 MB | Fastest. Needs piper binary (auto-downloaded). |
| Kokoro | High | ~326 MB | `pip install onda[kokoro]` |
| Orpheus | Expressive | ~3.2 GB | `pip install onda[orpheus]`. Emotion tags. Incomplete. |

All backends implement: `load(model_dir)` and `speak(text, output_path=None)`

## Model Storage

Models stored in `~/.onda/models/<model_name>/`

## Roadmap

**v0.1 — MVP**
- Model management (list, pull, remove, info)
- Plain text input: inline and --file .txt
- Cross-platform Piper binary auto-download (Mac arm64/x86, Windows, Linux)
- --out flag to save WAV

**v1.0 — Document support**
- --file input.pdf via optional dep: pymupdf or pdfplumber
- --file input.docx via optional dep: python-docx
- --start-page / --end-page for PDFs
- pip install onda[docs]

**v1.x — Polish**
- --speed flag
- --voice flag (Kokoro/Orpheus)
- MP3 output
- Progress bar for long files ("Chunk 4/23...")

## Optional Dependency Groups

```toml
[kokoro]   = kokoro, espeak-ng
[orpheus]  = llama-cpp-python, snac
[docs]     = pymupdf, python-docx
```

## Notes

- Do not auto-commit. Ask before any git operations.
- Orpheus backend is stubbed. snac token decoding not yet implemented.
- Piper binary auto-download must detect platform via platform.system() + platform.machine()
- chunk_text() splits at sentence boundaries, max 500 chars per chunk
- All backends must handle chunked input gracefully for long documents

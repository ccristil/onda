# onda

Ollama-style CLI for local TTS models. Pull, manage, and run open-source text-to-speech models from the command line. No cloud. No subscriptions.

```
$ onda run en_US-amy-low "Hello, world."
```

---

## Install

```bash
pip install onda
```

Or run without installing via `uvx`:

```bash
uvx onda run en_US-amy-low "Hello, world."
```

---

## Quickstart

```bash
# See available models
onda list

# Download a model
onda pull en_US-amy-low

# Set a default model (so you don't have to type it every time)
onda default en_US-amy-low

# Speak inline text (uses default model)
onda run "Hello, this is onda."

# Speak from a file
onda run --file notes.txt

# Save to WAV instead of playing
onda run "Save this." --out output.wav
```

---

## Commands

| Command | Description |
|---------|-------------|
| `onda list` | Show all models and download status |
| `onda pull <model>` | Download a model |
| `onda remove <model>` | Delete a downloaded model |
| `onda info <model>` | Show model details |
| `onda default <model>` | Set the default model |
| `onda default --show` | Print the current default model |
| `onda default --clear` | Remove the default model |
| `onda run [model] [text]` | Speak inline text (model optional if default is set) |
| `onda run [model] --file <path>` | Speak from a `.txt` file |
| `onda run ... --out <path>` | Save audio as WAV instead of playing |

---

## Models

| Name | Size | Description |
|------|------|-------------|
| `en_US-amy-low` | 5 MB | Fast and lightweight. Good for quick tasks. |
| `en_US-lessac-medium` | 63 MB | Balanced quality and speed. Recommended default. |
| `en_US-ryan-high` | 100 MB | High quality. Slower synthesis. |

Models are stored in `~/.onda/models/`.

---

## Backends

| Backend | Quality | Install |
|---------|---------|---------|
| Piper | Good | Included (default) |
| Kokoro | High | `pip install onda[kokoro]` |
| Orpheus | Expressive | `pip install onda[orpheus]` |

---

## Requirements

- Python 3.10+
- macOS (Linux/Windows coming soon)

---

## Roadmap

- **v0.1** — Model management + plain text input (current)
- **v1.0** — PDF and DOCX support (`onda[docs]`)
- **v1.x** — `--speed`, `--voice`, MP3 output, progress bar for long files

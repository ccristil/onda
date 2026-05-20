from __future__ import annotations

import re
import sounddevice as sd
import soundfile as sf
from pathlib import Path


def chunk_text(text: str, max_chars: int = 500) -> list[str]:
    """Split text at sentence boundaries, max_chars per chunk."""
    if not text.strip():
        return []

    # Split preserving the delimiter by splitting after sentence-ending punctuation + space
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue
        if len(sentence) > max_chars:
            # Hard-split an oversized single sentence
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i : i + max_chars])
        elif current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence

    if current:
        chunks.append(current)

    return chunks


def read_text_file(path: Path) -> str:
    """Read a .txt file. Raises SystemExit for unsupported types."""
    import typer
    if path.suffix.lower() != ".txt":
        typer.echo(
            f"Error: Unsupported file type '{path.suffix}'. Only .txt is supported in v0.1.",
            err=True,
        )
        raise SystemExit(1)
    return path.read_text(encoding="utf-8")


def fetch_url_text(url: str) -> str:
    """Fetch a URL and extract its main article text."""
    import trafilatura
    html = trafilatura.fetch_url(url)
    if html is None:
        raise RuntimeError("Could not fetch URL. Check your connection or the URL and try again.")
    text = trafilatura.extract(html)
    if text is None:
        raise RuntimeError("No readable content found. The page may be paywalled, require login, or have no extractable text.")
    return text.strip()


def play_wav(path: Path) -> None:
    """Play a WAV file synchronously."""
    data, samplerate = sf.read(str(path))
    sd.play(data, samplerate)
    sd.wait()

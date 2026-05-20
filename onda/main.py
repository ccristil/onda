from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(name="onda", help="Ollama-style CLI for local TTS models.")
console = Console()


@app.command(name="list")
def list_models() -> None:
    """Show available models and download status."""
    from onda.registry import load_registry, is_downloaded

    models = load_registry()
    table = Table("Name", "Backend", "Size (MB)", "Status", show_header=True)
    for m in models:
        status = "[green]✓ Downloaded[/]" if is_downloaded(m.name) else "– Not downloaded"
        table.add_row(m.name, m.backend, str(m.size_mb), status)
    console.print(table)


@app.command()
def pull(model: str = typer.Argument(..., help="Model name to download.")) -> None:
    """Download a model."""
    from onda.registry import get_model
    from onda.downloader import pull_model

    m = get_model(model)
    if m is None:
        raise typer.BadParameter(f"Model '{model}' not found. Run: onda list")
    pull_model(m)
    console.print(f"[green]✓[/] {model} downloaded successfully.")


@app.command()
def remove(model: str = typer.Argument(..., help="Model name to remove.")) -> None:
    """Delete a downloaded model."""
    from onda.registry import is_downloaded, model_dir

    if not is_downloaded(model):
        raise typer.BadParameter(f"Model '{model}' is not downloaded.")
    typer.confirm(f"Delete {model}?", abort=True)
    shutil.rmtree(model_dir(model))
    console.print(f"[green]✓[/] {model} removed.")


@app.command()
def info(model: str = typer.Argument(..., help="Model name to inspect.")) -> None:
    """Show model details."""
    from onda.registry import get_model, is_downloaded

    m = get_model(model)
    if m is None:
        raise typer.BadParameter(f"Model '{model}' not found. Run: onda list")
    status = "[green]Downloaded[/]" if is_downloaded(m.name) else "Not downloaded"
    content = (
        f"[bold]Backend:[/] {m.backend}\n"
        f"[bold]Size:[/] {m.size_mb} MB\n"
        f"[bold]Status:[/] {status}\n\n"
        f"{m.description}"
    )
    console.print(Panel(content, title=f"[bold]{m.name}[/]"))


@app.command()
def run(
    model: str = typer.Argument(..., help="Model name to use."),
    text: Optional[str] = typer.Argument(None, help="Text to speak."),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Text file to speak."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Save WAV instead of playing."),
) -> None:
    """Speak text using a local TTS model."""
    from onda.utils import read_text_file
    from onda.runner import run as runner_run

    if text and file:
        raise typer.BadParameter("Provide either inline text or --file, not both.")
    if not text and not file:
        raise typer.BadParameter("Provide text or --file input.")

    content = read_text_file(file) if file else text
    runner_run(model, content, output_path=out)

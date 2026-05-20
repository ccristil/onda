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
    from onda.config import get_default_model

    models = load_registry()
    current_default = get_default_model()
    table = Table("Name", "Backend", "Size (MB)", "Status", show_header=True)
    for m in models:
        status = "[green]✓ Downloaded[/]" if is_downloaded(m.name) else "– Not downloaded"
        name_cell = f"{m.name} [green](default)[/green]" if m.name == current_default else m.name
        table.add_row(name_cell, m.backend, str(m.size_mb), status)
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


@app.command(name="default")
def set_default(
    model: Optional[str] = typer.Argument(None, help="Model name to set as default."),
    show: bool = typer.Option(False, "--show", help="Print the current default model."),
    clear: bool = typer.Option(False, "--clear", help="Remove the default model."),
) -> None:
    """Get or set the default TTS model."""
    from onda.registry import get_model
    from onda.config import get_default_model, set_default_model, clear_default_model

    if sum([bool(model), show, clear]) > 1:
        raise typer.BadParameter("--show, --clear, and <model> are mutually exclusive.")

    if show or (model is None and not clear):
        current = get_default_model()
        console.print(f"Default model: [bold]{current}[/bold]" if current else "No default model set.")
        return

    if clear:
        clear_default_model()
        console.print("Default model cleared.")
        return

    if get_model(model) is None:
        raise typer.BadParameter(f"Model '{model}' not found in registry. Run: onda list")
    set_default_model(model)
    console.print(f"Default model set to: [bold]{model}[/bold]")


@app.command()
def run(
    model: Optional[str] = typer.Argument(None, help="Model name to use."),
    text: Optional[str] = typer.Argument(None, help="Text to speak."),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Text file to speak."),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="URL to fetch and read aloud."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Save WAV instead of playing."),
) -> None:
    """Speak text using a local TTS model."""
    from onda.utils import read_text_file
    from onda.runner import run as runner_run
    from onda.config import get_default_model

    if model is not None:
        from onda.registry import get_model as _get_model
        if _get_model(model) is None:
            # First arg isn't a model name — treat it as text, use default
            if text is not None:
                raise typer.BadParameter("Provide either inline text or --file, not both.")
            text = model
            model = None

    if model is None:
        model = get_default_model()
        if model is None:
            console.print(
                "[red]No model specified and no default set.[/red] "
                "Run: [bold]onda default <model>[/bold]"
            )
            raise typer.Exit(code=1)

    inputs = sum([bool(text), bool(file), bool(url)])
    if inputs > 1:
        console.print("[red]Please provide only one input: text, --file, or --url[/red]")
        raise typer.Exit(code=1)
    if inputs == 0:
        raise typer.BadParameter("Provide text, --file, or --url input.")

    if file:
        content = read_text_file(file)
        runner_run(model, content, output_path=out)
    elif url:
        import trafilatura
        try:
            with console.status("[bold green]Fetching URL...[/bold green]"):
                html = trafilatura.fetch_url(url)
                if html is None:
                    raise RuntimeError("Could not fetch URL. Check your connection or the URL and try again.")
            with console.status("[bold green]Extracting text...[/bold green]"):
                extracted = trafilatura.extract(html)
                if extracted is None:
                    raise RuntimeError("No readable content found. The page may be paywalled, require login, or have no extractable text.")
                content = extracted.strip()
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1)
        with console.status("[bold green]Rendering audio...[/bold green]"):
            runner_run(model, content, output_path=out)
    else:
        runner_run(model, text, output_path=out)

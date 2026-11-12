from __future__ import annotations

import json
from pathlib import Path

import typer

from .runtime import MemorySystemRuntime
from .scenarios import scenario_library


app = typer.Typer(help="Multi-context memory system MVP CLI")


@app.command("run-scenario")
def run_scenario(
    name: str = typer.Argument(..., help="Scenario name to execute"),
    store: Path = typer.Option(Path("memory_system.db"), "--store", help="SQLite store path"),
    config: Path | None = typer.Option(None, "--config", help="Optional config override YAML"),
) -> None:
    runtime = MemorySystemRuntime(config_path=str(config) if config else None, store_path=store)
    try:
        summary = runtime.run_scenario(name)
    finally:
        runtime.close()
    typer.echo(json.dumps(summary, indent=2))


@app.command("inspect")
def inspect(
    store: Path = typer.Option(Path("memory_system.db"), "--store", help="SQLite store path"),
    config: Path | None = typer.Option(None, "--config", help="Optional config override YAML"),
) -> None:
    runtime = MemorySystemRuntime(config_path=str(config) if config else None, store_path=store)
    try:
        typer.echo(json.dumps(runtime.inspect(), indent=2))
    finally:
        runtime.close()


@app.command("consolidate")
def consolidate(
    store: Path = typer.Option(Path("memory_system.db"), "--store", help="SQLite store path"),
    config: Path | None = typer.Option(None, "--config", help="Optional config override YAML"),
) -> None:
    runtime = MemorySystemRuntime(config_path=str(config) if config else None, store_path=store)
    try:
        typer.echo(json.dumps(runtime.consolidate_now(), indent=2))
    finally:
        runtime.close()


@app.command("list-scenarios")
def list_scenarios() -> None:
    typer.echo(json.dumps(sorted(scenario_library()), indent=2))


@app.command("dashboard")
def dashboard() -> None:
    """Launch the Streamlit GUI dashboard."""
    import subprocess
    import sys
    dashboard_path = str(Path(__file__).parent / "dashboard.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path], check=False)


def main() -> None:
    app()


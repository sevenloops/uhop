import click
from rich.console import Console
from uhop.optimizer import optimize
from uhop.hardware import get_hardware_summary

console = Console()

@click.group()
def main():
    """UHOP CLI — AI-Powered Universal Hardware Optimizer."""
    pass

@main.command()
def info():
    """Display detected hardware and backend availability."""
    console.print("[bold cyan]UHOP Hardware Report[/bold cyan]")
    console.print(get_hardware_summary())

@main.command()
@click.argument("function_path")
def optimize_func(function_path):
    """Optimize a Python function via @uhop.optimize"""
    console.print(f"[green]Optimizing function:[/green] {function_path}")
    # Import and trigger optimization decorator dynamically
    __import__(function_path)
    console.print("[bold green]Optimization complete![/bold green]")

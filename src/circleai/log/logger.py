import traceback
from typing import Callable, Any
from rich.console import Console

log_override: bool = False
console = Console()
printfn = console.print


def set_log_override(value: bool) -> None:
    global log_override, printfn
    log_override = value
    printfn = console.print if log_override else lambda _: None


def error(message: str) -> None:
    """Print error message in bold red."""
    printfn(f"[bold red]Error: {message}[/bold red]")


def warn(message: str) -> None:
    """Print warning message in bold yellow."""
    printfn(f"[bold yellow]Warning: {message}[/bold yellow]")


def info(message: str) -> None:
    """Print info message in bold blue."""
    printfn(f"[bold blue]Info: {message}[/bold blue]")


def success(message: str) -> None:
    """Print success message in bold green."""
    printfn(f"[bold green]Success: {message}[/bold green]")


def debug(message: str) -> None:
    """Print debug message in bold cyan."""
    printfn(f"[bold cyan]Debug: {message}[/bold cyan]")


def handle(message: str) -> None:
    """Print traceback and custom error message in bold red."""
    printfn(f"[bold red]{traceback.format_exc()}[/bold red]")
    error(message)


def check(condition: bool, message: str) -> None:
    """Check a condition and raise an error if it is false, displaying the error message in bold red."""
    check_err(condition, message, SystemExit)

def check_err(condition: bool, message: str, err: Exception) -> None:
    """Check a condition and raise an error if it is false, displaying the error message in bold red."""
    if not condition:
        error(message)
        raise err

def create_logger(
    fn: Callable[[Any], None], verbose: bool = False
) -> Callable[[Any], None]:
    """Create a logger function that calls fn if verbose is True, otherwise does nothing."""
    return fn if verbose else lambda _: None

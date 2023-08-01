# Copyright 2023 CircleML GitHub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import traceback
from typing import Callable, Any
from rich.console import Console

log_override: bool = False
console = Console()
printfn = console.print


def set_log_override(value: bool) -> None:
    global log_override, printfn
    log_override = value
    printfn = console.print if not log_override else lambda _: None


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

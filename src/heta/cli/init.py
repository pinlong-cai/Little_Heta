"""Interactive initialization wizard for Little Heta."""

from __future__ import annotations

import os
import pwd
import signal
from pathlib import Path
from typing import Callable

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from heta.assistants import install_assistant_skills, skill_template_hint
from heta.cli.branding import APP_TAGLINE, HETA, MUTED, OK, WARN, brand_line
from heta.config.io import CONFIG_PATH, save_config
from heta.config.schema import HetaConfig, InsertPlanningConfig, LLMConfig, MinerUConfig, VectorIndexConfig
from heta.providers.llm import validate_llm
from heta.providers.mineru import validate_mineru_cloud, validate_mineru_local

console = Console()

LLM_PROVIDERS = {1: "qwen", 2: "chatgpt", 3: "gemini"}
MINERU_OPTIONS = {1: "cloud", 2: "local", 3: "skip"}
MAX_RETRIES = 3


def interactive_init() -> None:
    """Run the Little Heta interactive initialization wizard."""
    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_cancel_signal)
    try:
        _run_interactive_init()
    except EOFError:
        console.print("\n[yellow]Initialization cancelled.[/yellow]")
        raise typer.Exit(130) from None
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)


def _run_interactive_init() -> None:
    _show_welcome()

    if CONFIG_PATH.exists():
        console.print(f"[{WARN}]?[/] Configuration already exists.")
        if not Confirm.ask("  Reinitialize?", default=False):
            console.print(f"[{MUTED}]Initialization cancelled.[/]")
            raise typer.Exit(0)

    llm_config = _configure_llm()
    partial_config = HetaConfig(
        version=1,
        llm=llm_config,
        mineru=MinerUConfig.disabled(),
        vector_index=VectorIndexConfig.enabled(),
        insert_planning=InsertPlanningConfig.enabled(),
    )
    save_config(partial_config)
    console.print(f"[{HETA}]→[/] wrote {CONFIG_PATH}")

    mineru_config = _configure_mineru()
    final_config = HetaConfig(
        version=1,
        llm=llm_config,
        mineru=mineru_config,
        vector_index=VectorIndexConfig.enabled(),
        insert_planning=InsertPlanningConfig.enabled(),
    )
    save_config(final_config)

    console.print(f"[{OK}]✓[/] little heta is ready")
    _install_assistant_skills()
    _show_summary(final_config)


def _handle_cancel_signal(signum: int, frame: object) -> None:
    console.print(f"\n[{WARN}]Initialization cancelled.[/]")
    raise typer.Exit(130)


def _show_welcome() -> None:
    console.print()
    console.print(
        Panel(
            f"{brand_line()}\n"
            f"[{MUTED}]{APP_TAGLINE}[/]\n\n"
            f"[{MUTED}]team:[/]      [bold]Knowledge[/][bold {HETA}]X[/][bold]Lab[/]\n"
            f"[{MUTED}]config:[/]    {_short_path(CONFIG_PATH)}",
            border_style=HETA,
            width=68,
            padding=(1, 2),
        )
    )
    console.print()
    console.print(f"  [{WARN}]Tip:[/] Run this once to connect Little Heta to your providers.")
    console.print()
    console.print(f"  [{MUTED}]Learn more:[/] https://github.com/KnowledgeXLab/Heta")
    console.print()
    console.print(f"[{MUTED}]$[/] [bold {HETA}]heta init[/]")
    console.print(f"[bold {HETA}]little heta setup[/]")


def _short_path(path: Path) -> str:
    try:
        home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, OSError):
        home = Path.home()
    try:
        text = "~/" + str(path.expanduser().resolve().relative_to(home.resolve()))
    except ValueError:
        text = str(path)
    except OSError:
        text = str(path)

    if len(text) <= 56:
        return text
    return text[:26].rstrip("/") + "/…/" + text[-26:].lstrip("/")


def _configure_llm() -> LLMConfig:
    console.print()
    console.print(f"[{WARN}]?[/] Choose provider")
    console.print(f"  [{HETA}]1[/] qwen")
    console.print(f"  [{HETA}]2[/] chatgpt")
    console.print(f"  [{HETA}]3[/] gemini")

    choice = _ask_choice("  Provider", LLM_PROVIDERS)
    provider = LLM_PROVIDERS[choice]

    api_key = _retry_secret(
        prompt="  Paste API key",
        validate=lambda key: validate_llm(provider, key),
        checking_message=f"Pinging {provider}",
        failure_message=(
            "Could not connect. The API key may be incorrect, "
            "or Little Heta could not reach the LLM provider."
        ),
        exhausted_message="LLM configuration failed. Initialization aborted.",
    )
    return LLMConfig(provider=provider, api_key=api_key)


def _configure_mineru() -> MinerUConfig:
    console.print()
    console.print(f"[{WARN}]?[/] Enable PDF and Office parsing with MinerU?")
    console.print(f"  [{HETA}]1[/] Cloud")
    console.print(f"  [{HETA}]2[/] Local sidecar")
    console.print(f"  [{HETA}]3[/] Skip for now")

    choice = _ask_choice("  MinerU option", MINERU_OPTIONS)
    option = MINERU_OPTIONS[choice]

    if option == "skip":
        return MinerUConfig.disabled()

    if option == "cloud":
        try:
            api_key = _retry_secret(
                prompt="  Paste MinerU API key",
                validate=validate_mineru_cloud,
                checking_message="Pinging MinerU Cloud",
                failure_message="Could not validate MinerU Cloud API key.",
                exhausted_message="MinerU not configured.",
                exit_on_exhausted=False,
            )
        except _RetryExhausted:
            return MinerUConfig.disabled()
        return MinerUConfig(enable=True, provider="cloud", api_key=api_key, endpoint=None)

    try:
        endpoint = _retry_text(
            prompt="  MinerU endpoint",
            default="http://127.0.0.1:8000",
            validate=validate_mineru_local,
            checking_message="Checking MinerU local health endpoint",
            failure_message="Could not connect to MinerU local endpoint.",
            exhausted_message="MinerU not configured.",
            exit_on_exhausted=False,
        )
    except _RetryExhausted:
        return MinerUConfig.disabled()
    return MinerUConfig(enable=True, provider="local", api_key=None, endpoint=endpoint.rstrip("/"))


def _ask_choice(label: str, choices: dict[int, str]) -> int:
    while True:
        choice = IntPrompt.ask(label)
        if choice in choices:
            return choice
        console.print(f"[{WARN}]?[/] Choose one of: {', '.join(map(str, choices))}")


def _retry_secret(
    *,
    prompt: str,
    validate: Callable[[str], bool],
    checking_message: str,
    failure_message: str,
    exhausted_message: str,
    exit_on_exhausted: bool = True,
) -> str:
    return _retry_value(
        prompt=prompt,
        default=None,
        password=True,
        validate=validate,
        checking_message=checking_message,
        failure_message=failure_message,
        exhausted_message=exhausted_message,
        exit_on_exhausted=exit_on_exhausted,
    )


def _retry_text(
    *,
    prompt: str,
    default: str | None,
    validate: Callable[[str], bool],
    checking_message: str,
    failure_message: str,
    exhausted_message: str,
    exit_on_exhausted: bool = True,
) -> str:
    return _retry_value(
        prompt=prompt,
        default=default,
        password=False,
        validate=validate,
        checking_message=checking_message,
        failure_message=failure_message,
        exhausted_message=exhausted_message,
        exit_on_exhausted=exit_on_exhausted,
    )


def _retry_value(
    *,
    prompt: str,
    default: str | None,
    password: bool,
    validate: Callable[[str], bool],
    checking_message: str,
    failure_message: str,
    exhausted_message: str,
    exit_on_exhausted: bool,
) -> str:
    attempts = 0
    while attempts < MAX_RETRIES:
        raw_value = Prompt.ask(prompt, default=default, password=password)
        value = (raw_value or "").strip()
        if not value:
            console.print(f"[{WARN}]?[/] Value cannot be empty.")
            continue

        attempts += 1
        with console.status(checking_message, spinner="dots"):
            ok = validate(value)

        if ok:
            console.print(f"[{OK}]✓[/] Provider reachable")
            return value

        remaining = MAX_RETRIES - attempts
        console.print(f"[{WARN}]?[/] {failure_message}")
        if remaining:
            console.print(f"[{MUTED}]  {remaining} attempt(s) remaining.[/]")

    console.print(f"[{WARN}]?[/] {exhausted_message}")
    if exit_on_exhausted:
        raise typer.Exit(1)
    raise _RetryExhausted


def _install_assistant_skills() -> None:
    """Install the Little Heta skill into supported AI coding assistants."""
    try:
        installed = install_assistant_skills()
    except Exception as exc:
        console.print(f"[{WARN}]?[/] Could not install assistant skills: {exc}")
        return

    console.print()
    console.print(f"[{OK}]✓[/] assistant skills installed")
    for item in installed:
        console.print(f"  [{MUTED}]{item.assistant}:[/] {_short_path(item.path)}")
    console.print(f"  [{MUTED}]Other agents:[/] {skill_template_hint()}.")


def _show_summary(config: HetaConfig) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style=f"bold {HETA}")
    table.add_column()
    table.add_row("config", str(CONFIG_PATH))
    table.add_row("provider", config.llm.provider)
    table.add_row("mineru docs", _mineru_summary(config.mineru))
    table.add_row("next", f"[bold {HETA}]heta insert ./notes[/] or [bold {HETA}]heta remember \"...\"[/]")

    console.print(
        Panel(
            table,
            title="ready",
            border_style=OK,
        )
    )


def _mineru_summary(config: MinerUConfig) -> str:
    if not config.enable:
        return "builtin parser fallback"
    if config.provider == "local":
        return f"local ({config.endpoint})"
    return "cloud"


class _RetryExhausted(Exception):
    pass


__all__ = ["interactive_init"]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Command lifecycle, discovery, and registration."""

from __future__ import annotations

import abc
import inspect
import logging
import re
import sys
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from types import ModuleType
from typing import Any, TextIO

from .component import ComponentSpec
from .context import ApplicationContext
from .invariants import (
    Invariant,
    InvariantDefinitionError,
    InvariantError,
    OptionCollisionError,
    PositionalDefinitionError,
    VariablePositionalInvariant,
)

_CAMEL_BOUNDARY = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)
_RESERVED_COMMANDS = {"help", "usage", "version"}
_COMMAND_TOKEN_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_LOG_LEVEL_ALIASES = {"WARN": "WARNING", "FATAL": "CRITICAL"}
_STANDARD_LOG_LEVELS = {
    logging.NOTSET,
    logging.DEBUG,
    logging.INFO,
    logging.WARNING,
    logging.ERROR,
    logging.CRITICAL,
}


class CommandError(RuntimeError):
    """A concise user-facing command failure."""


class CommandRegistrationError(RuntimeError):
    """Command names or aliases cannot be registered safely."""


class CommandCollisionError(CommandRegistrationError):
    """Two command classes use the same command token."""


def _write_line(stream: TextIO, message: object) -> None:
    text = str(message)
    stream.write(text)
    if not text.endswith("\n"):
        stream.write("\n")
    stream.flush()


def command_name(command_class: type[Command]) -> str:
    """Return the canonical kebab-case name for a command class."""

    override = command_class.__dict__.get("_name_")
    if override is not None:
        return str(override)
    class_name = command_class.__name__
    if not class_name.endswith("Command"):
        raise CommandRegistrationError(
            f"command class name must end in 'Command': {class_name}"
        )
    stem = class_name[: -len("Command")]
    return "-".join(part.lower() for part in _CAMEL_BOUNDARY.split(stem))


def command_shortname(
    command_class: type[Command],
    name: str | None = None,
) -> str | None:
    """Return an explicit or derived command alias."""

    canonical = name or command_name(command_class)
    for owner in command_class.__mro__:
        if "_shortname_" in owner.__dict__:
            value = owner.__dict__["_shortname_"]
            return None if value in {None, ""} else str(value)
        if owner is Command:
            break
    tokens = canonical.split("-")
    if len(tokens) == 1:
        return canonical
    return "".join(token[0] for token in tokens)


def command_description(command_class: type[Command]) -> str:
    """Return a concise description from command metadata or its docstring."""

    for attribute in ("_summary_", "_description_"):
        value = getattr(command_class, attribute, None)
        if value:
            return str(value)
    doc = inspect.getdoc(command_class) or ""
    return doc.split("\n", 1)[0]


def invariant_name(declaration_name: str) -> str:
    """Convert a nested ``*Arg`` or ``*Option`` name to snake case."""

    for suffix in ("Option", "Arg"):
        if declaration_name.endswith(suffix):
            stem = declaration_name[: -len(suffix)]
            return "_".join(
                part.lower() for part in _CAMEL_BOUNDARY.split(stem)
            )
    raise InvariantDefinitionError(
        f"invariant declaration must end in Arg or Option: {declaration_name}"
    )


def collect_invariants(
    command_class: type[Command],
) -> tuple[tuple[str, type[Invariant]], ...]:
    """Collect inherited invariants in stable declaration order."""

    declarations: dict[str, type[Invariant]] = {}
    for owner in reversed(command_class.__mro__):
        for nested_name, value in owner.__dict__.items():
            if not nested_name.endswith(("Arg", "Option")):
                continue
            if not inspect.isclass(value) or not issubclass(value, Invariant):
                continue
            name = invariant_name(nested_name)
            declarations[name] = value

    flags: dict[str, str] = {}
    saw_variable = False
    for name, declaration in declarations.items():
        declaration_flags = declaration.flags()
        if declaration_flags:
            for flag in declaration_flags:
                if not flag.startswith("-"):
                    raise InvariantDefinitionError(
                        f"invalid option flag {flag!r} on {name}"
                    )
                if flag in flags:
                    raise OptionCollisionError(
                        f"duplicate option flag {flag!r} on "
                        f"{flags[flag]} and {name}"
                    )
                flags[flag] = name
            continue
        if saw_variable:
            raise PositionalDefinitionError(
                "a variable positional invariant must be declared last"
            )
        if issubclass(declaration, VariablePositionalInvariant):
            if declaration._nargs not in {"?", "*", "+"}:
                raise PositionalDefinitionError(
                    f"invalid variable positional nargs: "
                    f"{declaration._nargs!r}"
                )
            saw_variable = True
    return tuple(declarations.items())


class Command(AbstractContextManager["Command"], abc.ABC):
    """Context-managed command with a deterministic execution lifecycle."""

    _log_level_ = True
    _quiet_ = False
    _verbose_ = False

    def __init__(
        self,
        *,
        input: TextIO | None = None,
        output: TextIO | None = None,
        error: TextIO | None = None,
        component: ComponentSpec | None = None,
        context: ApplicationContext | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "input",
            sys.stdin if input is None else input,
        )
        object.__setattr__(
            self,
            "output",
            sys.stdout if output is None else output,
        )
        object.__setattr__(
            self,
            "error",
            sys.stderr if error is None else error,
        )
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "arguments", [])
        object.__setattr__(self, "options", {})
        object.__setattr__(self, "result", None)
        object.__setattr__(self, "load_order", [])
        object.__setattr__(self, "_pending_values", {})
        object.__setattr__(self, "_exit_callbacks", [])
        object.__setattr__(self, "_chained_commands", [])
        object.__setattr__(self, "_lifecycle_state", "initialized")
        object.__setattr__(
            self, "_invariant_map", dict(collect_invariants(type(self)))
        )

    def __setattr__(self, name: str, value: Any) -> None:
        declarations = self.__dict__.get("_invariant_map", {})
        declaration = declarations.get(name)
        if declaration is not None:
            value = declaration.validate(value)
        object.__setattr__(self, name, value)

    def __enter__(self) -> Command:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._run_exit_callbacks()
        return False

    @property
    def name(self) -> str:
        return command_name(type(self))

    @property
    def shortname(self) -> str | None:
        return command_shortname(type(self), self.name)

    @property
    def lifecycle_state(self) -> str:
        return self._lifecycle_state

    def is_quiet(self) -> bool:
        return bool(getattr(self, "quiet", False))

    def bind(self, values: Mapping[str, Any]) -> Command:
        """Store parsed values for loading at the start of execution."""

        unknown = (
            set(values)
            - set(self._invariant_map)
            - {
                "log_level",
                "quiet",
                "verbose",
            }
        )
        if unknown:
            raise InvariantError(
                "unknown parsed value(s): " + ", ".join(sorted(unknown))
            )
        for name, declaration in self._invariant_map.items():
            if name in values:
                declaration.validate(values[name])
        if self._log_level_ and "log_level" in values:
            normalize_log_level(values["log_level"])
        object.__setattr__(self, "_pending_values", dict(values))
        return self

    def start(self, values: Mapping[str, Any] | None = None) -> Any:
        """Execute the lifecycle and return the command result."""

        try:
            if values is not None:
                self.bind(values)
            object.__setattr__(self, "_lifecycle_state", "loading")
            self._pre_load_options()
            self._load_options()
            object.__setattr__(self, "_lifecycle_state", "running")
            self._pre_run()
            result = self.run()
            if result is not None:
                object.__setattr__(self, "result", result)
            self._post_run()
            for command in tuple(self._chained_commands):
                command.start()
            object.__setattr__(self, "_lifecycle_state", "finished")
            return self.result
        finally:
            self._run_exit_callbacks()

    def _pre_load_options(self) -> None:
        """Hook called immediately before parsed values are loaded."""

    def _load_options(self) -> None:
        values = self._pending_values
        options: dict[str, Any] = {}
        arguments: list[Any] = []
        load_order: list[str] = []
        for name, declaration in self._invariant_map.items():
            value = (
                values[name]
                if name in values
                else declaration.default(type(self), name)
            )
            setattr(self, name, value)
            prepared = declaration.prepare(getattr(self, name))
            object.__setattr__(self, name, prepared)
            load_order.append(name)
            if declaration.flags():
                options[name] = prepared
            elif isinstance(prepared, list):
                arguments.extend(prepared)
            elif prepared is not None:
                arguments.append(prepared)

        if self._log_level_:
            level = values.get("log_level", "INFO")
            options["log_level"] = level
            self.set_log_level(level)
        if self._quiet_:
            quiet = bool(values.get("quiet", False))
            object.__setattr__(self, "quiet", quiet)
            options["quiet"] = quiet
        if self._verbose_:
            verbose = bool(values.get("verbose", False))
            object.__setattr__(self, "verbose", verbose)
            options["verbose"] = verbose

        object.__setattr__(self, "load_order", load_order)
        object.__setattr__(self, "options", options)
        object.__setattr__(self, "arguments", arguments)

    def _pre_run(self) -> None:
        """Hook called immediately before :meth:`run`."""

    @abc.abstractmethod
    def run(self) -> Any:
        """Perform domain work."""

    def _post_run(self) -> None:
        """Hook called after a successful :meth:`run`."""

    def _out(self, message: object = "") -> None:
        if not self.is_quiet():
            _write_line(self.output, message)

    def _err(self, message: object) -> None:
        _write_line(self.error, f"ERROR: {message}")

    def _warn(self, message: object) -> None:
        _write_line(self.error, f"WARNING: {message}")

    def _verbose(self, message: object) -> None:
        if bool(getattr(self, "verbose", False)) and not self.is_quiet():
            _write_line(self.output, message)

    def set_log_level(self, level: str | int) -> int:
        """Validate a standard log level and configure the component log."""

        numeric = normalize_log_level(level)
        object.__setattr__(self, "log_level", numeric)
        if self.context is not None:
            logger_name = (
                f"cuphoton.{self.component.group}"
                if self.component is not None
                else "cuphoton"
            )
            handler = configure_logging(
                self.context.log_file,
                numeric,
                logger=logging.getLogger(logger_name),
            )
            logger = logging.getLogger(logger_name)
            self.on_exit(handler.close)
            self.on_exit(logger.removeHandler, handler)
        return numeric

    def on_exit(
        self,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """Register a LIFO callback that runs when command execution ends."""

        self._exit_callbacks.append((callback, args, kwargs))
        return callback

    def _run_exit_callbacks(self) -> None:
        while self._exit_callbacks:
            callback, args, kwargs = self._exit_callbacks.pop()
            callback(*args, **kwargs)

    def prime(self, command_class: type[Command]) -> Command:
        """Create another command and copy values loaded by this command."""

        primed = command_class(
            input=self.input,
            output=self.output,
            error=self.error,
            component=self.component,
            context=self.context,
        )
        copied: dict[str, Any] = {}
        for name in self.load_order:
            if name in primed._invariant_map and hasattr(self, name):
                value = getattr(self, name)
                setattr(primed, name, value)
                copied[name] = value
        object.__setattr__(primed, "_pending_values", copied)
        return primed

    def stash(self, command: Command | type[Command]) -> Command:
        """Append an explicitly chained command for post-run execution."""

        chained = self.prime(command) if inspect.isclass(command) else command
        if not isinstance(chained, Command):
            raise TypeError("stash expects a Command instance or class")
        self._chained_commands.append(chained)
        return chained


class InvariantAwareCommand(Command):
    """Command base that binds nested declarative invariant classes."""


def normalize_log_level(level: str | int) -> int:
    """Normalize standard level names and exact standard numeric levels."""

    if isinstance(level, str):
        value = level.strip().upper()
        value = _LOG_LEVEL_ALIASES.get(value, value)
        if value.isdecimal():
            numeric = int(value)
        else:
            numeric = logging.getLevelNamesMapping().get(value, -1)
    else:
        numeric = int(level)
    if numeric not in _STANDARD_LOG_LEVELS:
        raise InvariantError(f"invalid log level: {level!r}")
    return numeric


def resolve_log_level(level: str | int) -> int:
    """Return the canonical numeric value for a standard logging level."""

    return normalize_log_level(level)


def configure_logging(
    log_file: str | Path,
    level: str | int = "INFO",
    *,
    logger: logging.Logger | None = None,
) -> logging.FileHandler:
    """Attach a formatted file handler and return it to the caller."""

    numeric = resolve_log_level(level)
    path = Path(log_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    target = logging.getLogger("cuphoton") if logger is None else logger
    target.setLevel(numeric)
    target.addHandler(handler)
    return handler


def discover_command_classes(module: ModuleType) -> tuple[type[Command], ...]:
    """Discover concrete, public command classes defined by ``module``."""

    discovered: list[type[Command]] = []
    tokens: dict[str, str] = {}
    for class_name, value in vars(module).items():
        if class_name.startswith("_") or not inspect.isclass(value):
            continue
        if value.__module__ != module.__name__:
            continue
        if not issubclass(value, Command) or inspect.isabstract(value):
            continue
        name = command_name(value)
        shortname = command_shortname(value, name)
        command_tokens = [name]
        if shortname is not None:
            command_tokens.append(shortname)
        for token in dict.fromkeys(command_tokens):
            if not _COMMAND_TOKEN_RE.fullmatch(token):
                raise CommandRegistrationError(
                    f"invalid command token {token!r} used by {class_name}"
                )
            if token in _RESERVED_COMMANDS:
                raise CommandRegistrationError(
                    f"reserved command token {token!r} used by {class_name}"
                )
            if token in tokens:
                raise CommandCollisionError(
                    f"duplicate command token {token!r} for "
                    f"{tokens[token]!r} and {name!r}"
                )
            tokens[token] = name
        collect_invariants(value)
        discovered.append(value)
    return tuple(discovered)


__all__ = [
    "Command",
    "CommandCollisionError",
    "CommandError",
    "CommandRegistrationError",
    "InvariantAwareCommand",
    "collect_invariants",
    "command_description",
    "command_name",
    "command_shortname",
    "configure_logging",
    "discover_command_classes",
    "normalize_log_level",
    "resolve_log_level",
]

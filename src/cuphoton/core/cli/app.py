# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reusable component parser and the public root router."""

from __future__ import annotations

import argparse
import importlib
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence, TextIO

from cuphoton import __version__

from .command import (
    Command,
    CommandError,
    command_description,
    command_name,
    command_shortname,
    discover_command_classes,
    normalize_log_level,
)
from .component import (
    COMPONENT_REGISTRY,
    COMPONENTS,
    ComponentSpec,
    get_component,
)
from .context import ApplicationContext
from .invariants import (
    BoolInvariant,
    CSVIntegerInvariant,
    CSVInvariant,
    CSVStringInvariant,
    FloatInvariant,
    IntegerInvariant,
    Invariant,
    InvariantError,
    NonNegativeIntegerInvariant,
    PairInvariant,
    PathValueInvariant,
    PositiveIntegerInvariant,
    SequenceInvariant,
    SetInvariant,
    VariablePositionalInvariant,
)


class _ParserFailure(Exception):
    def __init__(self, status: int, message: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class _ArgumentParser(argparse.ArgumentParser):
    def __init__(
        self,
        *args: Any,
        output: TextIO,
        error_stream: TextIO,
        capitalized_usage: bool,
        **kwargs: Any,
    ) -> None:
        self.output_stream = output
        self.error_stream = error_stream
        self.capitalized_usage = capitalized_usage
        super().__init__(*args, **kwargs)

    def exit(self, status: int = 0, message: str | None = None) -> None:
        raise _ParserFailure(status, message)

    def error(self, message: str) -> None:
        self.print_usage(self.error_stream)
        self._print_message(
            f"{self.prog}: error: {message}\n",
            self.error_stream,
        )
        raise _ParserFailure(2)

    def print_help(self, file: TextIO | None = None) -> None:
        super().print_help(file or self.output_stream)

    def format_usage(self) -> str:
        usage = super().format_usage()
        if self.capitalized_usage:
            return usage.replace("usage:", "Usage:", 1)
        return usage

    def format_help(self) -> str:
        help_text = super().format_help()
        if self.capitalized_usage:
            return help_text.replace("usage:", "Usage:", 1)
        return help_text


class _InvariantType:
    def __init__(
        self,
        declaration: type[Invariant],
        *,
        item_only: bool = False,
    ) -> None:
        self.declaration = declaration
        self.item_only = item_only

    def __call__(self, value: str) -> Any:
        try:
            if self.item_only:
                converter = self.declaration._item_type or str
                return converter(value)
            return self.declaration.validate(value)
        except (InvariantError, TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class CommandLine:
    """One registered canonical command and its optional alias."""

    name: str
    shortname: str
    command_class: type[Command]
    description: str


class ComponentCLI:
    """Reusable parser and dispatcher for one component."""

    def __init__(
        self,
        component: ComponentSpec,
        command_classes: Sequence[type[Command]],
        *,
        input: TextIO,
        out: TextIO,
        err: TextIO,
        environ: Mapping[str, str] | None,
        program_name: str | None = None,
        context: ApplicationContext | None = None,
    ) -> None:
        self.component = component
        self.input = input
        self.out = out
        self.err = err
        self.environ = environ
        self.context = context
        self.program_name = program_name or component.program_name
        self._commands_by_name: dict[str, CommandLine] = {}
        self._commands_by_token: dict[str, CommandLine] = {}
        for command_class in command_classes:
            name = command_name(command_class)
            alias = command_shortname(command_class, name)
            exposed_alias = alias or name
            commandline = CommandLine(
                name=name,
                shortname=exposed_alias,
                command_class=command_class,
                description=command_description(command_class),
            )
            self._commands_by_name[name] = commandline
            self._commands_by_token[name] = commandline
            if alias is not None:
                self._commands_by_token[alias] = commandline

    @property
    def command_names(self) -> tuple[str, ...]:
        return tuple(self._commands_by_name)

    def run(self, argv: Sequence[str] | None = None) -> int:
        args = list(sys.argv[1:] if argv is None else argv)
        if (
            not args
            or args in (["help"], ["usage"])
            or args[0]
            in {
                "-h",
                "--help",
            }
        ):
            self.print_help()
            return 0
        if (
            len(args) == 1
            and args[0] in {"-v", "-V", "--version", "version"}
            and self.component.supports_version_command
        ):
            self.print_version()
            return 0
        if args[0] in {"help", "usage"}:
            if len(args) == 1:
                self.print_help()
                return 0
            return self.print_command_help(args[1])

        commandline = self._commands_by_token.get(args[0])
        if commandline is None:
            self._error(f"Unknown command '{args[0]}'")
            return 2
        if any(token in {"-h", "--help"} for token in args[1:]):
            self._parser(commandline).print_help()
            return 0

        parser = self._parser(commandline)
        try:
            if self.component.parser_style == "optparse":
                namespace, extras = parser.parse_known_args(args[1:])
                if extras:
                    self._error("invalid number of arguments")
                    return 1
            else:
                namespace = parser.parse_args(args[1:])
        except _ParserFailure as exc:
            if exc.message:
                self.err.write(exc.message)
            return int(exc.status)

        command = commandline.command_class(
            input=self.input,
            output=self.out,
            error=self.err,
            component=self.component,
            context=(
                self.context
                if self.context is not None
                else ApplicationContext.for_component(
                    self.component,
                    self.environ,
                )
            ),
        )
        try:
            with redirect_stdout(self.out), redirect_stderr(self.err):
                command.start(vars(namespace))
        except CommandError as exc:
            command._err(exc)
            return 1
        except InvariantError as exc:
            command._err(exc)
            return 1
        except SystemExit as exc:
            if isinstance(exc.code, int):
                return exc.code
            if exc.code:
                command._err(exc.code)
                return 1
            return 0
        except Exception as exc:
            command._err(exc)
            return 1
        return 0

    def print_version(self) -> None:
        self.out.write(_component_version(self.component))

    def print_help(self) -> None:
        heading = (
            "Available commands"
            if self.component.group_help_style == "commands"
            else "Available subcommands"
        )
        self.out.write(f"{self.program_name}\n\n")
        description = (
            self.component.group_help_description
            or self.component.description
        )
        self.out.write(f"{description}\n\n")
        self.out.write(f"{heading}:\n")
        for commandline in self._commands_by_name.values():
            alias = ""
            if (
                commandline.shortname != commandline.name
                or self.component.show_equal_aliases
            ):
                alias = f" ({commandline.shortname})"
            indent = " " * (2 + self.component.command_indent)
            self.out.write(
                f"{indent}{commandline.name}{alias}"
                f"{'  ' if commandline.description else ''}"
                f"{commandline.description}\n"
            )
        if self.component.supports_version_command:
            self.out.write("  version  Show the component version.\n")
        self.out.write(
            f"\nUse '{self.program_name} help COMMAND' for command help.\n"
        )

    def print_command_help(self, token: str) -> int:
        commandline = self._commands_by_token.get(token)
        if commandline is None:
            self._error(f"Unknown subcommand '{token}'")
            return 1
        self._parser(commandline).print_help()
        return 0

    def _error(self, message: object) -> None:
        self.err.write(f"ERROR: {message}\n")

    def _parser(self, commandline: CommandLine) -> _ArgumentParser:
        parser = _ArgumentParser(
            prog=f"{self.program_name} {commandline.name}",
            description=commandline.description or None,
            add_help=False,
            output=self.out,
            error_stream=self.err,
            capitalized_usage=self.component.parser_style == "optparse",
            formatter_class=argparse.HelpFormatter,
        )
        for name, declaration in commandline.command_class(
            output=self.out,
            error=self.err,
        )._invariant_map.items():
            self._add_invariant(
                parser,
                commandline.command_class,
                name,
                declaration,
            )
        if commandline.command_class._log_level_:
            parser.add_argument(
                "--log-level",
                default="INFO",
                type=self._log_level,
                metavar="LEVEL",
                help="Python logging level. [default: INFO]",
            )
        if commandline.command_class._quiet_:
            parser.add_argument(
                "-q",
                "--quiet",
                action="store_true",
                default=False,
                help="Suppress normal output.",
            )
        if commandline.command_class._verbose_:
            parser.add_argument(
                "-v",
                "--verbose",
                action="store_true",
                default=False,
                help="Enable verbose output.",
            )
        return parser

    @staticmethod
    def _log_level(value: str) -> str:
        try:
            normalize_log_level(value)
        except InvariantError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
        return value

    def _add_invariant(
        self,
        parser: _ArgumentParser,
        command_class: type[Command],
        name: str,
        declaration: type[Invariant],
    ) -> None:
        flags = declaration.flags()
        positional = not flags
        target = [name] if positional else list(flags)
        kwargs: dict[str, Any] = {
            "help": self._help_text(
                declaration,
                name,
                declaration.default(command_class, name),
            ),
        }
        default = declaration.default(command_class, name)
        if not positional:
            kwargs["dest"] = name
            kwargs["required"] = declaration.required()
            kwargs["default"] = default

        if declaration._metavar is not None:
            kwargs["metavar"] = declaration._metavar
        elif flags and not issubclass(declaration, BoolInvariant):
            if self.component.parser_style == "optparse":
                kwargs["metavar"] = name.upper()
            else:
                metavar_flag = next(
                    (
                        flag
                        for flag in reversed(flags)
                        if flag.startswith("--")
                    ),
                    flags[-1],
                )
                kwargs["metavar"] = (
                    metavar_flag.lstrip("-")
                    .replace(
                        "-",
                        "_",
                    )
                    .upper()
                )
        if issubclass(declaration, BoolInvariant):
            kwargs["action"] = declaration._action or "store_true"
        elif issubclass(declaration, PairInvariant):
            kwargs["nargs"] = 2
            kwargs["type"] = _InvariantType(declaration, item_only=True)
        elif issubclass(declaration, SequenceInvariant):
            kwargs["action"] = declaration._action or "append"
            kwargs["type"] = _InvariantType(declaration, item_only=True)
        elif issubclass(declaration, VariablePositionalInvariant):
            kwargs["nargs"] = declaration._nargs
            kwargs["type"] = _InvariantType(declaration, item_only=True)
        elif issubclass(declaration, SetInvariant):
            kwargs["choices"] = sorted(declaration._set)
            if declaration._action is not None:
                kwargs["action"] = declaration._action
        elif issubclass(
            declaration,
            (PositiveIntegerInvariant, NonNegativeIntegerInvariant),
        ):
            kwargs["type"] = (
                _InvariantType(declaration)
                if self.component.parser_style == "argparse"
                else int
            )
        elif issubclass(declaration, IntegerInvariant):
            kwargs["type"] = int
        elif issubclass(declaration, FloatInvariant):
            kwargs["type"] = float
        elif issubclass(
            declaration,
            (CSVInvariant, CSVStringInvariant, CSVIntegerInvariant),
        ):
            kwargs["type"] = _InvariantType(declaration)
        elif issubclass(declaration, PathValueInvariant):
            kwargs["type"] = Path
        if positional:
            kwargs.pop("help", None)
        action = parser.add_argument(*target, **kwargs)
        if positional:
            action.required = declaration.required()

    @staticmethod
    def _help_text(
        declaration: type[Invariant],
        name: str,
        default: Any,
    ) -> str:
        text = declaration._help or name.replace("_", " ")
        if default is not None:
            text = text.replace("%default", str(default))
        return text


def build_component_cli(
    component: str | ComponentSpec,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
    program_name: str | None = None,
    command_module: ModuleType | None = None,
    context: ApplicationContext | None = None,
    istream: TextIO | None = None,
    ostream: TextIO | None = None,
    estream: TextIO | None = None,
) -> ComponentCLI:
    """Load one commands module and build a reusable component CLI."""

    spec = get_component(component)
    input_stream = sys.stdin if istream is None else istream
    output = _stream(out, ostream, sys.stdout, "out", "ostream")
    error = _stream(err, estream, sys.stderr, "err", "estream")
    module = (
        importlib.import_module(spec.module_name)
        if command_module is None
        else command_module
    )
    return ComponentCLI(
        spec,
        discover_command_classes(module),
        input=input_stream,
        out=output,
        err=error,
        environ=environ,
        program_name=program_name,
        context=context,
    )


def run_component(
    component: str | ComponentSpec,
    argv: Sequence[str] | None = None,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
    program_name: str | None = None,
    command_module: ModuleType | None = None,
    context: ApplicationContext | None = None,
    istream: TextIO | None = None,
    ostream: TextIO | None = None,
    estream: TextIO | None = None,
) -> int:
    """Build and run one public or external component."""

    spec = get_component(component)
    args = list(sys.argv[1:] if argv is None else argv)
    output = _stream(out, ostream, sys.stdout, "out", "ostream")
    error = _stream(err, estream, sys.stderr, "err", "estream")
    if len(args) == 1 and args[0] in {"-v", "-V", "--version", "version"}:
        if spec.supports_version_command:
            output.write(_component_version(spec))
            return 0
        error.write(f"ERROR: Unknown command '{args[0]}'\n")
        return 2
    return build_component_cli(
        spec,
        out=output,
        err=error,
        environ=environ,
        program_name=program_name,
        command_module=command_module,
        context=context,
        istream=istream,
    ).run(args)


def main(
    argv: Sequence[str] | None = None,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
    program_name: str = "cuphoton",
) -> int:
    """Route the root executable to one of the six public components."""

    args = list(sys.argv[1:] if argv is None else argv)
    output = sys.stdout if out is None else out
    error = sys.stderr if err is None else err
    if not args or args in (["help"], ["--help"], ["-h"]):
        _print_root_help(output, program_name)
        return 0
    if len(args) == 1 and args[0] in {"-V", "--version", "version"}:
        output.write(f"{__version__}\n")
        return 0
    if args[0] in {"-V", "--version", "version"}:
        error.write(f"ERROR: {args[0]} does not accept arguments\n")
        return 2
    routed = args
    if args[0] == "help" and len(args) > 1:
        routed = [args[1], "help", *args[2:]]
    spec = COMPONENT_REGISTRY.get(routed[0])
    if spec is None:
        error.write(f"ERROR: Unknown component group '{routed[0]}'\n")
        return 2
    return run_component(
        spec,
        routed[1:],
        out=output,
        err=error,
        environ=environ,
        program_name=f"{program_name} {spec.group}",
    )


def _stream(
    short: TextIO | None,
    legacy: TextIO | None,
    default: TextIO,
    short_name: str,
    legacy_name: str,
) -> TextIO:
    if short is not None and legacy is not None and short is not legacy:
        raise TypeError(
            f"{short_name} and {legacy_name} refer to different streams"
        )
    return short if short is not None else legacy or default


def _component_version(component: ComponentSpec) -> str:
    label = (
        component.group
        if component.version_label is None
        else component.version_label
    )
    return f"{label + ' ' if label else ''}{__version__}\n"


def _print_root_help(out: TextIO, program_name: str) -> None:
    out.write(f"{program_name}\n\n")
    out.write("GPU-accelerated astronomy and imaging tools.\n\n")
    out.write("Available groups:\n")
    for component in COMPONENTS:
        out.write(f"  {component.group}  {component.description}\n")
    out.write(f"\nUse '{program_name} GROUP --help' for group help.\n")


__all__ = [
    "CommandLine",
    "CLI",
    "ComponentCLI",
    "build_component_cli",
    "main",
    "run_component",
]

CLI = ComponentCLI

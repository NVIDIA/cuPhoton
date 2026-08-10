# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import logging
import types
from pathlib import Path

import pytest

from cuphoton import __version__
from cuphoton.core.cli import (
    CLI,
    ApplicationContext,
    CommandCollisionError,
    CommandLine,
    ComponentSpec,
    InvariantAwareCommand,
    OptionCollisionError,
    PositionalDefinitionError,
    PositiveIntegerInvariant,
    StringInvariant,
    VariablePositionalInvariant,
    build_component_cli,
    collect_invariants,
    configure_logging,
    discover_command_classes,
    main,
    resolve_log_level,
    run_component,
)


@pytest.mark.parametrize("token", ["-V", "--version", "version"])
def test_root_version_spellings_are_exact(token: str) -> None:
    output = io.StringIO()
    error = io.StringIO()

    assert main([token], out=output, err=error) == 0
    assert output.getvalue() == f"{__version__}\n"
    assert error.getvalue() == ""

    output.seek(0)
    output.truncate()
    assert main([token, "extra"], out=output, err=error) == 2
    assert output.getvalue() == ""


@pytest.mark.parametrize("token", ["-v", "-V", "--version", "version"])
def test_group_version_forms_do_not_construct_a_parser(
    token: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cuphoton.core.cli import app

    def unexpected_build(*args: object, **kwargs: object) -> None:
        raise AssertionError("version routing reached parser construction")

    monkeypatch.setattr(app, "build_component_cli", unexpected_build)
    expected = {
        "xfit": f"{__version__}\n",
        "xpois": f"{__version__}\n",
        "xscan": f"{__version__}\n",
        "xrep": f"{__version__}\n",
        "xray": f"cuphoton xray {__version__}\n",
    }
    for group, text in expected.items():
        output = io.StringIO()
        assert run_component(group, [token], out=output) == 0
        assert output.getvalue() == text

    error = io.StringIO()
    assert run_component("xdr", [token], err=error) == 2
    assert token in error.getvalue()


def test_root_help_can_delegate_through_a_group() -> None:
    output = io.StringIO()
    error = io.StringIO()

    assert (
        main(
            ["help", "xray", "doctor"],
            out=output,
            err=error,
            program_name="custom-tool",
        )
        == 0
    )
    assert "usage: custom-tool xray doctor" in output.getvalue()
    assert error.getvalue() == ""


def _memory_commands() -> types.ModuleType:
    module = types.ModuleType("memory_contract_commands")
    module.__dict__.update(
        {
            "InvariantAwareCommand": InvariantAwareCommand,
            "StringInvariant": StringInvariant,
        }
    )
    exec(
        """
class CopyCommand(InvariantAwareCommand):
    class PrefixArg(StringInvariant):
        _arg = "--prefix"
        _default = "read"

    def run(self):
        self._out(f"{self.prefix}:{self.input.read()}")
""",
        module.__dict__,
    )
    return module


def test_legacy_builder_keywords_preserve_objects(tmp_path: Path) -> None:
    spec = ComponentSpec("addon", "cuphoton.addon", "In-memory commands.")
    context = ApplicationContext(
        component=spec,
        config_home=tmp_path / "config",
        state_home=tmp_path / "state",
        data_home=tmp_path / "data",
        log_file_override=tmp_path / "chosen" / "addon.log",
    )
    input_stream = io.StringIO("payload")
    output_stream = io.StringIO()
    error_stream = io.StringIO()
    module = _memory_commands()

    cli = build_component_cli(
        spec,
        command_module=module,
        context=context,
        istream=input_stream,
        ostream=output_stream,
        estream=error_stream,
        program_name="addon-cli",
    )

    assert isinstance(cli, CLI)
    assert cli.context is context
    assert cli.input is input_stream
    assert cli.out is output_stream
    assert cli.err is error_stream
    assert cli.run(["copy", "--prefix", "copied"]) == 0
    assert output_stream.getvalue() == "copied:payload\n"
    assert error_stream.getvalue() == ""
    assert context.log_file == (tmp_path / "chosen" / "addon.log").resolve()


def test_run_component_accepts_an_in_memory_module(tmp_path: Path) -> None:
    spec = ComponentSpec("addon2", "cuphoton.addon2", "In-memory commands.")
    context = ApplicationContext(
        spec,
        tmp_path / "config",
        tmp_path / "state",
        tmp_path / "data",
    )
    output = io.StringIO()

    assert (
        run_component(
            spec,
            ["copy"],
            command_module=_memory_commands(),
            context=context,
            istream=io.StringIO("value"),
            ostream=output,
        )
        == 0
    )
    assert output.getvalue() == "read:value\n"


def test_stream_aliases_reject_different_targets() -> None:
    spec = ComponentSpec("addon3", "cuphoton.addon3", "In-memory commands.")
    with pytest.raises(TypeError, match="out and ostream"):
        build_component_cli(
            spec,
            command_module=_memory_commands(),
            out=io.StringIO(),
            ostream=io.StringIO(),
        )


def test_facade_exposes_specific_definition_error_categories() -> None:
    assert CommandLine.__name__ == "CommandLine"

    class BadOptionsCommand(InvariantAwareCommand):
        class OneArg(StringInvariant):
            _arg = "--duplicate"

        class TwoArg(StringInvariant):
            _arg = "--duplicate"

        def run(self) -> None:
            return None

    with pytest.raises(OptionCollisionError):
        collect_invariants(BadOptionsCommand)

    class BadPositionalsCommand(InvariantAwareCommand):
        class TailArg(VariablePositionalInvariant):
            pass

        class LaterArg(StringInvariant):
            pass

        def run(self) -> None:
            return None

    with pytest.raises(PositionalDefinitionError):
        collect_invariants(BadPositionalsCommand)

    module = types.ModuleType("colliding_contract_commands")
    module.__dict__["InvariantAwareCommand"] = InvariantAwareCommand
    exec(
        """
class FirstThingCommand(InvariantAwareCommand):
    _shortname_ = "same"
    def run(self):
        return None

class SecondThingCommand(InvariantAwareCommand):
    _shortname_ = "same"
    def run(self):
        return None
""",
        module.__dict__,
    )
    with pytest.raises(CommandCollisionError):
        discover_command_classes(module)


def test_public_logging_helpers_create_only_the_parent(
    tmp_path: Path,
) -> None:
    log_file = tmp_path / "nested" / "component.log"
    logger = logging.getLogger("cuphoton.contract-test")
    handler = configure_logging(log_file, "warn", logger=logger)
    try:
        assert resolve_log_level("FATAL") == logging.CRITICAL
        assert logger.level == logging.WARNING
        assert log_file.parent.is_dir()
        assert log_file.is_file()
    finally:
        logger.removeHandler(handler)
        handler.close()


def test_optparse_bounds_are_loaded_as_domain_values() -> None:
    module = types.ModuleType("bounded_contract_commands")
    module.__dict__.update(
        {
            "InvariantAwareCommand": InvariantAwareCommand,
            "PositiveIntegerInvariant": PositiveIntegerInvariant,
        }
    )
    exec(
        """
class CheckCommand(InvariantAwareCommand):
    class CountArg(PositiveIntegerInvariant):
        _arg = "--count"
        _required = True

    def run(self):
        return None
""",
        module.__dict__,
    )
    output = io.StringIO()
    error = io.StringIO()
    spec = ComponentSpec(
        "bounded",
        "cuphoton.bounded",
        "Bounded values.",
        parser_style="optparse",
    )

    assert (
        run_component(
            spec,
            ["check", "--count", "0"],
            command_module=module,
            out=output,
            err=error,
        )
        == 1
    )
    assert "at least 1" in error.getvalue()

    argparse_spec = ComponentSpec(
        "bounded2",
        "cuphoton.bounded2",
        "Bounded values.",
        parser_style="argparse",
    )
    error = io.StringIO()
    assert (
        run_component(
            argparse_spec,
            ["check", "--count", "0"],
            command_module=module,
            err=error,
        )
        == 2
    )
    assert "at least 1" in error.getvalue()


def test_context_defaults_are_resolved_without_creation(
    tmp_path: Path,
) -> None:
    context = ApplicationContext(
        "xscan",
        tmp_path / "cfg",
        tmp_path / "state",
        tmp_path / "data",
    )

    assert context.component.group == "xscan"
    assert context.config_dir == (tmp_path / "cfg/cuphoton/xscan").resolve()
    assert context.workspace_dir == context.state_dir
    assert context.runs_dir == context.state_dir / "runs"
    assert context.log_file == context.logs_dir / "xscan.log"
    assert not context.config_dir.exists()
    assert not context.state_dir.exists()

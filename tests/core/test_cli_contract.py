# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import io
import logging
import sys
import types
from pathlib import Path

import pytest

from cuphoton.core.cli import (
    COMPONENTS,
    ApplicationContext,
    BoolInvariant,
    CommandError,
    CommandRegistrationError,
    ComponentSpec,
    CSVIntegerInvariant,
    ExistingPathInvariant,
    InvariantAwareCommand,
    InvariantDefinitionError,
    InvariantError,
    MkDirectoryInvariant,
    NonNegativeIntegerInvariant,
    OutPathInvariant,
    PairInvariant,
    SequenceInvariant,
    SetInvariant,
    StringInvariant,
    VariablePositionalInvariant,
    build_component_cli,
    collect_invariants,
    command_name,
    command_shortname,
    discover_command_classes,
    get_component,
    main,
    normalize_log_level,
    run_component,
)


def test_public_command_surface_counts_are_exact() -> None:
    per_group: list[tuple[str, int, int, int, int]] = []
    for component in COMPONENTS:
        module = importlib.import_module(component.module_name)
        command_classes = discover_command_classes(module)
        aliases = sum(
            command_shortname(command_class, command_name(command_class))
            is not None
            for command_class in command_classes
        )
        arguments = sum(
            len(collect_invariants(command_class))
            for command_class in command_classes
        )
        per_group.append(
            (
                component.group,
                len(command_classes),
                aliases,
                arguments,
                int(component.supports_version_command),
            )
        )

    assert per_group == [
        ("xdr", 1, 0, 13, 0),
        ("xfit", 3, 3, 17, 1),
        ("xpois", 6, 6, 95, 1),
        ("xscan", 42, 42, 166, 1),
        ("xrep", 6, 6, 101, 1),
        ("xray", 33, 31, 374, 1),
    ]
    assert len(per_group) == 6
    assert sum(item[1] for item in per_group) == 91
    assert sum(item[1] + item[4] for item in per_group) == 96
    assert sum(item[2] for item in per_group) == 88
    assert sum(item[3] for item in per_group) == 766


def test_public_registry_order_and_component_derivations() -> None:
    assert tuple(item.group for item in COMPONENTS) == (
        "xdr",
        "xfit",
        "xpois",
        "xscan",
        "xrep",
        "xray",
    )
    spec = get_component("xrep")
    assert spec.module_name == "cuphoton.xrep.commands"
    assert spec.program_name == "cuphoton xrep"
    assert spec.app_dir == "xrep"
    assert spec.log_env == "CUPHOTON_XREP_LOG_FILE"
    assert spec.default_log_filename == "xrep.log"


@pytest.mark.parametrize("group", ["Upper", "two-words", "1x", ""])
def test_component_rejects_invalid_group_tokens(group: str) -> None:
    with pytest.raises(ValueError):
        ComponentSpec(group, "cuphoton.extra", "external")


def test_external_component_is_preserved_and_context_is_side_effect_free(
    tmp_path: Path,
) -> None:
    spec = ComponentSpec(
        "extra7",
        "cuphoton.extra",
        "External commands.",
        log_filename="external.log",
    )
    environ = {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "cfg"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        spec.log_env: str(tmp_path / "elsewhere" / "chosen.log"),
    }

    assert get_component(spec) is spec
    context = ApplicationContext.for_component(spec, environ)
    assert (
        context.config_dir
        == (tmp_path / "cfg" / "cuphoton" / "extra7").resolve()
    )
    assert context.workspace_dir == context.state_dir
    assert context.runs_dir == context.state_dir / "runs"
    assert context.logs_dir == context.state_dir / "logs"
    assert (
        context.log_file == (tmp_path / "elsewhere" / "chosen.log").resolve()
    )
    assert not (tmp_path / "cfg").exists()
    assert not (tmp_path / "state").exists()


def test_root_streams_and_lazy_imports() -> None:
    out = io.StringIO()
    err = io.StringIO()
    before = {
        name
        for name in sys.modules
        if name.startswith(
            (
                "cuphoton.xdr",
                "cuphoton.xfit",
                "cuphoton.xpois",
                "cuphoton.xscan",
                "cuphoton.xrep",
                "cuphoton.xray",
            )
        )
    }

    assert main([], out=out, err=err) == 0
    assert "xdr" in out.getvalue()
    assert err.getvalue() == ""
    assert {
        name
        for name in sys.modules
        if name.startswith(
            (
                "cuphoton.xdr",
                "cuphoton.xfit",
                "cuphoton.xpois",
                "cuphoton.xscan",
                "cuphoton.xrep",
                "cuphoton.xray",
            )
        )
    } == before

    out.seek(0)
    out.truncate()
    assert main(["not-a-group"], out=out, err=err) == 2
    assert out.getvalue() == ""
    assert "not-a-group" in err.getvalue()


def test_supported_version_does_not_load_the_commands_module() -> None:
    spec = ComponentSpec(
        "lazy1",
        "cuphoton.lazy",
        "Lazy external component.",
        commands_module="contract_missing_commands",
    )
    out = io.StringIO()

    assert run_component(spec, ["version"], out=out) == 0
    assert "contract_missing_commands" not in sys.modules
    assert out.getvalue().startswith("lazy1 ")


class _LifecycleCommand(InvariantAwareCommand):
    events: list[str] = []
    count = None

    class CountArg(NonNegativeIntegerInvariant):
        _arg = "--count"
        _default = 3

    def _pre_load_options(self) -> None:
        self.events.append("pre-load")

    def _load_options(self) -> None:
        super()._load_options()
        self.events.append(f"load:{self.count}")

    def _pre_run(self) -> None:
        self.events.append("pre-run")

    def run(self) -> str:
        self.events.append("run")
        return "result"

    def _post_run(self) -> None:
        self.events.append("post-run")


def test_command_lifecycle_and_exit_callbacks_are_ordered() -> None:
    events: list[str] = []
    command = _LifecycleCommand()
    command.events = events
    command.on_exit(events.append, "exit-first")
    command.on_exit(events.append, "exit-last")

    assert command.start({"count": "0", "log_level": "INFO"}) == "result"
    assert events == [
        "pre-load",
        "load:0",
        "pre-run",
        "run",
        "post-run",
        "exit-last",
        "exit-first",
    ]
    assert command.lifecycle_state == "finished"


def test_declared_assignment_validates_immediately() -> None:
    command = _LifecycleCommand()

    command.count = "4"
    assert command.count == 4
    with pytest.raises(InvariantError, match="non-negative"):
        command.count = -1


def test_prime_copies_falsey_values() -> None:
    source = _LifecycleCommand()
    source.load_order = ["count"]
    source.count = 0

    primed = source.prime(_LifecycleCommand)

    assert primed.count == 0


def test_invariant_shapes_and_definition_checks(tmp_path: Path) -> None:
    class ShapeCommand(InvariantAwareCommand):
        mode = None
        pair = None
        item = None
        tail = None

        class ModeArg(SetInvariant):
            _arg = "--mode"
            _set = {"a", "b"}

        class PairArg(PairInvariant):
            _arg = "--pair"
            _item_type = int

        class ItemArg(SequenceInvariant):
            _arg = "--item"
            _item_type = Path

        class TailArg(VariablePositionalInvariant):
            _nargs = "*"

        def run(self) -> None:
            return None

    names = tuple(name for name, _ in collect_invariants(ShapeCommand))
    assert names == ("mode", "pair", "item", "tail")
    command = ShapeCommand()
    command.mode = "a"
    command.pair = ["2", "5"]
    command.item = [tmp_path / "one", tmp_path / "two"]
    command.tail = ["x", "y"]
    assert command.pair == (2, 5)
    assert command.item == [tmp_path / "one", tmp_path / "two"]
    assert command.tail == ["x", "y"]
    with pytest.raises(InvariantError, match="one of"):
        command.mode = "c"

    class DuplicateFlagsCommand(InvariantAwareCommand):
        class FirstArg(StringInvariant):
            _arg = "--same"

        class SecondArg(StringInvariant):
            _arg = "--same"

        def run(self) -> None:
            return None

    with pytest.raises(InvariantDefinitionError, match="duplicate"):
        collect_invariants(DuplicateFlagsCommand)


def test_path_preparation_occurs_only_during_loading(tmp_path: Path) -> None:
    class PathsCommand(InvariantAwareCommand):
        directory = None
        output = None

        class DirectoryArg(MkDirectoryInvariant):
            _arg = "--directory"

        class OutputArg(OutPathInvariant):
            _arg = "--output"

        def run(self) -> None:
            return None

    directory = tmp_path / "made"
    output = tmp_path / "nested" / "result.txt"
    command = PathsCommand()
    command.directory = directory
    command.output = output
    assert not directory.exists()
    assert not output.parent.exists()

    command.start(
        {
            "directory": directory,
            "output": output,
            "log_level": "INFO",
        }
    )
    assert directory.is_dir()
    assert output.parent.is_dir()
    assert not output.exists()


def test_existing_path_invariant_checks_filesystem(tmp_path: Path) -> None:
    assert ExistingPathInvariant.validate(str(tmp_path)) == str(tmp_path)
    with pytest.raises(InvariantError, match="does not exist"):
        ExistingPathInvariant.validate(str(tmp_path / "missing"))


def test_csv_integer_validation_preserves_the_raw_value() -> None:
    assert CSVIntegerInvariant.validate("1, 2,3") == "1, 2,3"
    with pytest.raises(InvariantError, match="invalid item"):
        CSVIntegerInvariant.validate("1,two")


def test_log_levels_accept_names_aliases_and_standard_numbers() -> None:
    assert normalize_log_level("NOTSET") == logging.NOTSET
    assert normalize_log_level("debug") == logging.DEBUG
    assert normalize_log_level("WARN") == logging.WARNING
    assert normalize_log_level("fatal") == logging.CRITICAL
    assert normalize_log_level(logging.ERROR) == logging.ERROR
    with pytest.raises(InvariantError, match="invalid log level"):
        normalize_log_level(15)


def _external_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    namespace = module.__dict__
    namespace.update(
        {
            "BoolInvariant": BoolInvariant,
            "CommandError": CommandError,
            "InvariantAwareCommand": InvariantAwareCommand,
            "StringInvariant": StringInvariant,
        }
    )
    exec(
        """
class EchoCommand(InvariantAwareCommand):
    value = None
    upper = False

    class ValueArg(StringInvariant):
        _arg = "--value"
        _required = True

    class UpperArg(BoolInvariant):
        _arg = "--upper"

    def run(self):
        if self.value == "fail":
            raise CommandError("requested failure")
        payload = self.value.upper() if self.upper else self.value
        self._out(payload)
""",
        namespace,
    )
    return module


def test_external_component_build_parse_execute_and_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "contract_external_commands"
    module = _external_module(module_name)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec = ComponentSpec(
        "extra2",
        "cuphoton.extra",
        "External command fixture.",
        commands_module=module_name,
        parser_style="argparse",
    )
    out = io.StringIO()
    err = io.StringIO()
    cli = build_component_cli(
        spec,
        out=out,
        err=err,
        environ={"HOME": "/tmp"},
    )

    assert cli.command_names == ("echo",)
    assert cli.run(["echo", "--value", "mixed", "--upper"]) == 0
    assert out.getvalue() == "MIXED\n"

    out.seek(0)
    out.truncate()
    assert cli.run(["echo", "--value", "fail"]) == 1
    assert out.getvalue() == ""
    assert "requested failure" in err.getvalue()


def test_discovery_rejects_reserved_command_tokens() -> None:
    module = types.ModuleType("reserved_contract_commands")
    namespace = module.__dict__
    namespace["InvariantAwareCommand"] = InvariantAwareCommand
    exec(
        """
class HelpCommand(InvariantAwareCommand):
    def run(self):
        return None
""",
        namespace,
    )

    with pytest.raises(CommandRegistrationError, match="reserved"):
        discover_command_classes(module)

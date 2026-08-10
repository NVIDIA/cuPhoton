# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Declarative value conversion and validation for command classes."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

_MISSING = object()


class InvariantError(ValueError):
    """A user-provided value does not satisfy an invariant."""


class InvariantDefinitionError(TypeError):
    """An invariant declaration is internally inconsistent."""


class OptionCollisionError(InvariantDefinitionError):
    """Two invariant declarations use the same option flag."""


class PositionalDefinitionError(InvariantDefinitionError):
    """Positional invariant declarations cannot be parsed safely."""


class Invariant:
    """Base declaration for one parsed command value."""

    _arg: str | tuple[str, ...] | None = None
    _help: str | None = None
    _metavar: str | tuple[str, ...] | None = None
    _default: Any = _MISSING
    _mandatory = False
    _required = False
    _action: str | None = None
    _nargs: str | int | None = None
    _item_type: type[Any] | None = None
    _parser_error: str | None = None
    _type_desc = "value"

    @classmethod
    def validate(cls, value: Any) -> Any:
        """Validate and convert one value."""

        if value is None:
            return None
        if cls._item_type is not None:
            try:
                return cls._item_type(value)
            except (TypeError, ValueError) as exc:
                cls._invalid(value, str(exc))
        return value

    @classmethod
    def prepare(cls, value: Any) -> Any:
        """Perform loading-time filesystem preparation, if any."""

        return value

    @classmethod
    def default(cls, owner: type[Any], name: str) -> Any:
        if cls._default is not _MISSING:
            return copy.deepcopy(cls._default)
        return copy.deepcopy(getattr(owner, name, None))

    @classmethod
    def flags(cls) -> tuple[str, ...]:
        if cls._arg is None:
            return ()
        if isinstance(cls._arg, tuple):
            return cls._arg
        return tuple(part for part in cls._arg.split("/") if part)

    @classmethod
    def required(cls) -> bool:
        return bool(cls._mandatory or cls._required)

    @classmethod
    def _invalid(cls, value: Any, detail: str | None = None) -> None:
        if cls._parser_error:
            raise InvariantError(cls._parser_error)
        expected = getattr(cls, "expected", None)
        message = f"invalid {cls._type_desc}: {value!r}"
        if expected:
            message = f"{message}; expected {expected}"
        elif detail:
            message = f"{message} ({detail})"
        raise InvariantError(message)


class BoolInvariant(Invariant):
    """Boolean switch invariant."""

    _default = False
    _action = "store_true"

    @classmethod
    def validate(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value in {0, 1}:
            return bool(value)
        cls._invalid(value, "must be a boolean")


class StringInvariant(Invariant):
    """String value with inclusive length bounds."""

    _minlen = 1
    _maxlen: int | None = 1024
    _type_desc = "string"

    @classmethod
    def validate(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value)
        if cls._minlen is not None and len(text) < cls._minlen:
            raise InvariantError(
                f"must contain at least {cls._minlen} character(s)"
            )
        if cls._maxlen is not None and len(text) > cls._maxlen:
            raise InvariantError(
                f"must contain at most {cls._maxlen} character(s)"
            )
        return text


class IntegerInvariant(Invariant):
    """Integer value with inclusive numeric bounds."""

    _min: int | None = None
    _max: int | None = None
    _type_desc = "integer"

    @classmethod
    def validate(cls, value: Any) -> int | None:
        if value is None:
            return None
        try:
            converted = int(value)
        except (TypeError, ValueError):
            cls._invalid(value, "must be an integer")
        cls._check_bounds(converted)
        return converted

    @classmethod
    def _check_bounds(cls, value: int) -> None:
        if cls._min is not None and value < cls._min:
            raise InvariantError(f"must be at least {cls._min}")
        if cls._max is not None and value > cls._max:
            raise InvariantError(f"must be at most {cls._max}")


class PositiveIntegerInvariant(IntegerInvariant):
    """Strictly positive integer."""

    _min = 1


class NonNegativeIntegerInvariant(IntegerInvariant):
    """Integer greater than or equal to zero."""

    _min = 0

    @classmethod
    def _check_bounds(cls, value: int) -> None:
        if value < 0:
            raise InvariantError("must be non-negative")
        super()._check_bounds(value)


class FloatInvariant(Invariant):
    """Floating-point value with inclusive numeric bounds."""

    _min: float | None = None
    _max: float | None = None
    _type_desc = "number"

    @classmethod
    def validate(cls, value: Any) -> float | None:
        if value is None:
            return None
        try:
            converted = float(value)
        except (TypeError, ValueError):
            cls._invalid(value, "must be a number")
        if cls._min is not None and converted < cls._min:
            raise InvariantError(f"must be at least {cls._min}")
        if cls._max is not None and converted > cls._max:
            raise InvariantError(f"must be at most {cls._max}")
        return converted


class SetInvariant(Invariant):
    """String selected from a fixed set."""

    _set: set[Any] | frozenset[Any] = frozenset()

    @classmethod
    def validate(cls, value: Any) -> Any:
        if value is None:
            return None
        if cls._action == "append":
            items = value if isinstance(value, (list, tuple)) else [value]
            invalid = [item for item in items if item not in cls._set]
            if invalid:
                choices = ", ".join(str(item) for item in sorted(cls._set))
                raise InvariantError(f"must be one of: {choices}")
            return list(items)
        if value not in cls._set:
            choices = ", ".join(str(item) for item in sorted(cls._set))
            raise InvariantError(f"must be one of: {choices}")
        return value


class MultipleSetInvariant(SetInvariant):
    """Comma-separated subset of a fixed set."""

    @classmethod
    def validate(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value)
        items = [item.strip() for item in text.split(",") if item.strip()]
        if not items:
            cls._invalid(value, "must contain at least one value")
        invalid = [item for item in items if item not in cls._set]
        if invalid:
            choices = ", ".join(str(item) for item in sorted(cls._set))
            raise InvariantError(f"must be a subset of: {choices}")
        return text


class CSVInvariant(Invariant):
    """Base comma-separated value invariant."""

    _item_type: type[Any] = str

    @classmethod
    def validate(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            tokens = list(value)
            text = ",".join(str(item) for item in tokens)
        else:
            text = str(value)
            tokens = [
                item.strip() for item in text.split(",") if item.strip()
            ]
        if not tokens:
            cls._invalid(value, "must contain at least one item")
        try:
            for item in tokens:
                cls._item_type(item)
        except (TypeError, ValueError):
            cls._invalid(value, "contains an invalid item")
        return text


class CSVStringInvariant(Invariant):
    """Comma-separated string values."""

    _item_type = str
    validate = classmethod(CSVInvariant.validate.__func__)


class CSVIntegerInvariant(CSVStringInvariant):
    """Comma-separated integer values."""

    _item_type = int


class SequenceInvariant(StringInvariant):
    """Repeatable option whose item conversion is applied per occurrence."""

    _action = "append"
    _item_type: type[Any] = str

    @classmethod
    def validate(cls, value: Any) -> list[Any] | None:
        if value is None:
            return None
        items = value if isinstance(value, (list, tuple)) else [value]
        try:
            converted = []
            for item in items:
                StringInvariant.validate.__func__(cls, item)
                converted.append(cls._item_type(item))
            return converted
        except (TypeError, ValueError):
            cls._invalid(value, "contains an invalid item")


class PairInvariant(SequenceInvariant):
    """Option that consumes exactly two values."""

    _action = None
    _nargs = 2
    _item_type: type[Any] = str

    @classmethod
    def validate(cls, value: Any) -> tuple[Any, Any] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            cls._invalid(value, "must contain exactly two values")
        try:
            StringInvariant.validate.__func__(cls, value[0])
            StringInvariant.validate.__func__(cls, value[1])
            return (cls._item_type(value[0]), cls._item_type(value[1]))
        except (TypeError, ValueError):
            cls._invalid(value, "contains an invalid item")


class PositionalInvariant(StringInvariant):
    """Fixed positional command value."""

    _required = True


class VariablePositionalInvariant(StringInvariant):
    """Variable-cardinality positional command values."""

    _nargs: str = "*"
    _required = False

    @classmethod
    def validate(cls, value: Any) -> list[Any] | None:
        if value is None:
            return None
        items = value if isinstance(value, (list, tuple)) else [value]
        converter = cls._item_type or str
        try:
            return [converter(item) for item in items]
        except (TypeError, ValueError):
            cls._invalid(value, "contains an invalid item")


class PathInvariant(StringInvariant):
    """Path-like string invariant."""

    _type_desc = "path"
    _maxlen = 4096


class PathValueInvariant(StringInvariant):
    """Path invariant that stores a :class:`pathlib.Path`."""

    _type_desc = "path"
    _maxlen = 4096

    @classmethod
    def validate(cls, value: Any) -> Path | None:
        if value is None:
            return None
        text = super().validate(value)
        assert text is not None
        return Path(text).expanduser()


class ExistingPathInvariant(StringInvariant):
    """Path that must exist."""

    _type_desc = "existing path"
    _maxlen = 4096

    @classmethod
    def validate(cls, value: Any) -> Any:
        converted = super().validate(value)
        if (
            converted is not None
            and not Path(converted).expanduser().exists()
        ):
            raise InvariantError(f"path does not exist: {converted}")
        return converted


class DirectoryInvariant(ExistingPathInvariant):
    """Path that must identify an existing directory."""

    @classmethod
    def validate(cls, value: Any) -> Any:
        converted = super().validate(value)
        if (
            converted is not None
            and not Path(converted).expanduser().is_dir()
        ):
            raise InvariantError(f"path is not a directory: {converted}")
        return converted


class MkDirectoryInvariant(PathValueInvariant):
    """Directory path created only when command values are loaded."""

    @classmethod
    def prepare(cls, value: Any) -> Any:
        if value is not None:
            Path(value).mkdir(parents=True, exist_ok=True)
        return value


class OutPathInvariant(PathValueInvariant):
    """Output path whose parent is created while loading values."""

    @classmethod
    def prepare(cls, value: Any) -> Any:
        if value is not None:
            Path(value).parent.mkdir(parents=True, exist_ok=True)
        return value


__all__ = [
    "BoolInvariant",
    "CSVIntegerInvariant",
    "CSVInvariant",
    "CSVStringInvariant",
    "DirectoryInvariant",
    "ExistingPathInvariant",
    "FloatInvariant",
    "IntegerInvariant",
    "Invariant",
    "InvariantDefinitionError",
    "InvariantError",
    "MkDirectoryInvariant",
    "MultipleSetInvariant",
    "NonNegativeIntegerInvariant",
    "OutPathInvariant",
    "OptionCollisionError",
    "PairInvariant",
    "PathInvariant",
    "PathValueInvariant",
    "PositiveIntegerInvariant",
    "PositionalInvariant",
    "PositionalDefinitionError",
    "SequenceInvariant",
    "SetInvariant",
    "StringInvariant",
    "VariablePositionalInvariant",
]

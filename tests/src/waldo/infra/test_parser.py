r"""
Parser unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Union, get_origin

import numpy as np
import pytest
import torch

from waldo.infra.parser import _is_union, build_with_type_check, get_jsonl_keys, json_serializable, load_jsonl_to_numpy


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------
@dataclass
class SimpleConfig:
    name: str
    value: int


@dataclass
class NestedConfig:
    inner: SimpleConfig
    count: int


@dataclass
class OptionalConfig:
    name: str
    value: int | None = None


@dataclass
class LiteralConfig:
    mode: Literal["train", "test"]


@dataclass
class AnyConfig:
    value: Any


@dataclass
class UnionConfig:
    value: int | None


@dataclass
class UnionStrIntConfig:
    value: str | int


@dataclass
class UnionIntFloatConfig:
    value: int | float


# -------------------------------------------------------------------
# Trivial cases
# -------------------------------------------------------------------
class TestTrivialCases:
    r"""Tests for trivial early-exit cases."""

    def test_none_data_returns_none(self) -> None:
        r"""None data should be returned as-is regardless of type."""
        assert build_with_type_check(int, None) is None

    def test_any_type_returns_data_as_is(self) -> None:
        r"""Any field typed as Any should return data without conversion."""
        result = build_with_type_check(AnyConfig, {"value": {"key": "value"}})
        assert result.value == {"key": "value"}

    def test_already_correct_type_returns_as_is(self) -> None:
        r"""Data already of the correct type should be returned as-is."""
        assert build_with_type_check(int, 42) == 42

    def test_already_correct_str_type_returns_as_is(self) -> None:
        r"""String data already of the correct type should be returned as-is."""
        assert build_with_type_check(str, "hello") == "hello"


# -------------------------------------------------------------------
# Primitive types
# -------------------------------------------------------------------
class TestPrimitiveTypes:
    r"""Tests for primitive type conversion."""

    def test_int_from_string(self) -> None:
        r"""Integer should be built from a string."""
        assert build_with_type_check(int, "42") == 42

    def test_float_from_string(self) -> None:
        r"""Float should be built from a string."""
        assert build_with_type_check(float, "3.14") == pytest.approx(3.14)

    def test_str_from_int(self) -> None:
        r"""String should be built from an integer."""
        assert build_with_type_check(str, 42) == "42"

    def test_bool_from_int(self) -> None:
        r"""Bool should be built from an integer."""
        assert build_with_type_check(bool, 1) is True

    def test_invalid_primitive_raises_value_error(self) -> None:
        r"""Invalid primitive conversion should raise a ValueError."""
        with pytest.raises(ValueError):
            build_with_type_check(int, [1, 2, 3])


# -------------------------------------------------------------------
# Dataclass
# -------------------------------------------------------------------
class TestDataclass:
    r"""Tests for dataclass construction."""

    def test_simple_dataclass(self) -> None:
        r"""Simple dataclass should be built from a dict."""
        result = build_with_type_check(SimpleConfig, {"name": "test", "value": 42})
        assert isinstance(result, SimpleConfig)
        assert result.name == "test"
        assert result.value == 42

    def test_nested_dataclass(self) -> None:
        r"""Nested dataclass should be built recursively."""
        result = build_with_type_check(
            NestedConfig,
            {
                "inner": {"name": "test", "value": 42},
                "count": 3,
            },
        )
        assert isinstance(result, NestedConfig)
        assert isinstance(result.inner, SimpleConfig)
        assert result.inner.name == "test"
        assert result.count == 3

    def test_extra_fields_are_ignored(self) -> None:
        r"""Extra fields not in the dataclass should be silently ignored."""
        result = build_with_type_check(SimpleConfig, {"name": "test", "value": 42, "extra": "ignored"})
        assert isinstance(result, SimpleConfig)
        assert not hasattr(result, "extra")

    def test_missing_fields_use_defaults(self) -> None:
        r"""Missing optional fields should use their default values."""
        result = build_with_type_check(OptionalConfig, {"name": "test"})
        assert isinstance(result, OptionalConfig)
        assert result.value is None


# -------------------------------------------------------------------
# List
# -------------------------------------------------------------------
class TestList:
    r"""Tests for list construction."""

    def test_list_of_ints(self) -> None:
        r"""List of ints should be built from a list of strings."""
        result = build_with_type_check(list[int], ["1", "2", "3"])
        assert result == [1, 2, 3]

    def test_list_of_strings(self) -> None:
        r"""List of strings should be built correctly."""
        result = build_with_type_check(list[str], ["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_empty_list(self) -> None:
        r"""Empty list should return an empty list."""
        result = build_with_type_check(list[int], [])
        assert result == []


# -------------------------------------------------------------------
# Dict
# -------------------------------------------------------------------
class TestDict:
    r"""Tests for dict construction."""

    def test_dict_of_str_to_int(self) -> None:
        r"""Dict of str to int should be built correctly."""
        result = build_with_type_check(dict[str, int], {"a": "1", "b": "2"})
        assert result == {"a": 1, "b": 2}

    def test_empty_dict(self) -> None:
        r"""Empty dict should return an empty dict."""
        result = build_with_type_check(dict[str, int], {})
        assert result == {}


# -------------------------------------------------------------------
# Union
# -------------------------------------------------------------------
class TestUnion:
    r"""Tests for union type handling."""

    def test_union_int_or_none_with_int(self) -> None:
        r"""Union of int or None should return int when given an int."""
        result = build_with_type_check(UnionConfig, {"value": 42})
        assert result.value == 42

    def test_union_int_or_none_with_none(self) -> None:
        r"""Union of int or None should return None when given None."""
        result = build_with_type_check(UnionConfig, {"value": None})
        assert result.value is None

    def test_union_str_or_int_with_str(self) -> None:
        r"""Union of str or int should return str when given a string."""
        result = build_with_type_check(UnionStrIntConfig, {"value": "hello"})
        assert result.value == "hello"

    def test_union_no_match_raises_value_error(self) -> None:
        r"""Union with no matching type should raise a ValueError."""
        with pytest.raises((ValueError, TypeError)):
            build_with_type_check(UnionIntFloatConfig, {"value": "not_a_number"})


# -------------------------------------------------------------------
# Literal
# -------------------------------------------------------------------
class TestLiteral:
    r"""Tests for Literal type handling."""

    def test_valid_literal(self) -> None:
        r"""Valid literal value should be returned as-is."""
        result = build_with_type_check(LiteralConfig, {"mode": "train"})
        assert result.mode == "train"

    def test_invalid_literal_raises_value_error(self) -> None:
        r"""Invalid literal value should raise a ValueError."""
        with pytest.raises(ValueError, match="not a valid literal"):
            build_with_type_check(LiteralConfig, {"mode": "val"})


# -------------------------------------------------------------------
# List and dict type checking
# -------------------------------------------------------------------
class TestListAndDictTypeCheck:
    r"""Tests verifying that list and dict items are type-checked recursively."""

    def test_list_items_are_converted(self) -> None:
        r"""List items should be converted to the declared element type."""
        result = build_with_type_check(list[int], ["1", "2", "3"])
        assert all(isinstance(x, int) for x in result)

    def test_dict_values_are_converted(self) -> None:
        r"""Dict values should be converted to the declared value type."""
        result = build_with_type_check(dict[str, int], {"a": "1", "b": "2"})
        assert all(isinstance(v, int) for v in result.values())


# -------------------------------------------------------------------
# _is_union
# -------------------------------------------------------------------
class TestIsUnion:
    r"""Tests for _is_union helper."""

    def test_native_union(self) -> None:
        r"""Native X | Y union syntax should be recognized as a union."""
        assert _is_union(get_origin(int | str))

    def test_typing_union(self) -> None:
        r"""typing.Union should be recognized as a union."""
        assert _is_union(get_origin(Union[int, str]))  # noqa: UP007

    def test_non_union(self) -> None:
        r"""Non-union type should not be recognized as a union."""
        assert not _is_union(list)


# -------------------------------------------------------------------
# Torch dtype
# -------------------------------------------------------------------
class TestDtype:
    r"""Tests for torch dtypes built from configuration files."""

    @pytest.mark.parametrize("name", ["bfloat16", "torch.bfloat16"])
    def test_dtype_from_name(self, name: str) -> None:
        r"""A dtype name, with or without the torch prefix, should be converted to the torch dtype."""
        assert build_with_type_check(torch.dtype, name) is torch.bfloat16

    def test_dtype_is_returned_as_is(self) -> None:
        r"""A torch dtype should be returned unchanged."""
        assert build_with_type_check(torch.dtype, torch.float16) is torch.float16

    @pytest.mark.parametrize("name", ["unexisting", "nn"])
    def test_invalid_dtype_raises_value_error(self, name: str) -> None:
        r"""A name that is not a torch dtype should raise a ValueError, even if it is a torch attribute."""
        with pytest.raises(ValueError, match="Could not initialize"):
            build_with_type_check(torch.dtype, name)


# -------------------------------------------------------------------
# JSONL loading utilities
# -------------------------------------------------------------------
@pytest.fixture
def jsonl_path(tmp_path: Path) -> Path:
    r"""A jsonl file with a missing key, a malformed line and a key only present in the last line."""
    path = tmp_path / "data.jsonl"
    lines = [json.dumps({"a": 1, "b": 2.0}), json.dumps({"a": 3}), "not json", json.dumps({"a": 5, "c": "x"})]
    path.write_text("\n".join(lines) + "\n")
    return path


class TestJsonl:
    r"""Tests for jsonl loading utilities."""

    def test_keys_of_all_lines(self, jsonl_path: Path) -> None:
        r"""With readall, keys should be collected over all valid lines."""
        assert sorted(get_jsonl_keys(str(jsonl_path))) == ["a", "b", "c"]

    def test_keys_of_first_line(self, jsonl_path: Path) -> None:
        r"""Without readall, only the keys of the first line should be returned."""
        assert sorted(get_jsonl_keys(str(jsonl_path), readall=False)) == ["a", "b"]

    def test_load_to_numpy(self, jsonl_path: Path) -> None:
        r"""Malformed lines should be skipped and missing values replaced by None."""
        data = load_jsonl_to_numpy(str(jsonl_path))
        np.testing.assert_array_equal(data["a"], [1, 3, 5])
        assert data["b"].tolist() == [2.0, None, None]
        assert data["c"].tolist() == [None, None, "x"]

    def test_load_selected_keys(self, jsonl_path: Path) -> None:
        r"""Only the requested keys should be loaded."""
        assert list(load_jsonl_to_numpy(str(jsonl_path), keys=["a"])) == ["a"]


# -------------------------------------------------------------------
# Object type
# -------------------------------------------------------------------
class TestJsonSerializable:
    r"""Tests for json_serializable."""

    def test_paths_and_devices_are_converted_to_strings(self) -> None:
        r"""Paths and torch devices should be converted to strings, other values kept as they are."""
        config = {"path": Path("/tmp/runs"), "device": torch.device("cpu"), "seed": 42, "name": "vit"}
        result = json_serializable(config)
        assert result == {"path": "/tmp/runs", "device": "cpu", "seed": 42, "name": "vit"}
        json.dumps(result)

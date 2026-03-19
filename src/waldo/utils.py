r"""
Utils functions for typing and creating configuration files.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
import types
from dataclasses import fields, is_dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration type
# -----------------------------------------------------------------------------


def _build_dataclass(object_type: type, data: dict[str, Any]) -> Any:
    hints = get_type_hints(object_type)
    remaining = dict(data)
    field_values = {}
    for data_field in fields(object_type):
        if not data_field.init:
            continue
        fname = data_field.name
        ftype = hints[fname]
        if fname in remaining:
            field_values[fname] = build_with_type_check(ftype, remaining.pop(fname))
        else:
            logger.debug(f"Field '{fname}' not found in {object_type}.")
    for fname in remaining:
        logger.warning(f"Field '{fname}' ignored when initializing {object_type}.")
    return object_type(**field_values)


def _build_list(args: tuple, data: Any) -> list:
    return [build_with_type_check(args[0], item) for item in data]


def _build_dict(args: tuple, data: Any) -> dict:
    return {build_with_type_check(args[0], k): build_with_type_check(args[1], v) for k, v in data.items()}


def _build_union(args: tuple, data: Any) -> Any:
    for arg in args:
        try:
            return build_with_type_check(arg, data)
        except (TypeError, ValueError):
            continue
    raise ValueError(f"Could not match value '{data}' to any type in union {args}.")


def _build_literal(args: tuple, data: Any) -> Any:
    if data not in args:
        raise ValueError(f"Value '{data}' is not a valid literal. Expected one of {args}.")
    return data


def _build_primitive(object_type: type, data: Any) -> Any:
    try:
        return object_type(data)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Could not initialize {object_type} with value '{data}'.") from err


def _is_union(origin: Any) -> bool:
    r"""
    Check whether a type origin is a union type.

    This handles both ``typing.Union[X, Y]`` (pre Python 3.10) and ``X | Y``
    (native union syntax introduced in Python 3.10), as ``typing.get_origin``
    returns ``typing.Union`` for the former and ``types.UnionType`` for the
    latter without normalization.
    """
    return origin is Union or origin is types.UnionType


def build_with_type_check(object_type: type, data: Any) -> Any:
    """Recursively initializes a typed object from a nested dictionary."""
    # Trivial cases
    if data is None or object_type is Any:
        return data

    args = get_args(object_type)
    origin = get_origin(object_type)

    if is_dataclass(object_type):
        return _build_dataclass(object_type, data)
    if origin is list and len(args) == 1:
        return _build_list(args, data)
    if origin is dict and len(args) == 2:
        return _build_dict(args, data)
    if _is_union(origin):
        return _build_union(args, data)
    if origin is Literal:
        return _build_literal(args, data)

    # Already the right type
    base_type = origin or object_type
    if isinstance(base_type, type) and isinstance(data, base_type):
        return data

    return _build_primitive(object_type, data)

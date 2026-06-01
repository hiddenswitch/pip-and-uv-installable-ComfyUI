from __future__ import annotations

import dataclasses
import decimal
import typing
from collections.abc import Mapping

from comfy.api.exceptions import ApiTypeError, ApiValueError


class immutabledict(Mapping):
    """Small immutable mapping compatible with the old generated schema output."""

    _dict: dict

    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __contains__(self, key):
        return key in self._dict

    def __repr__(self):
        return f"{self.__class__.__name__}({self._dict!r})"

    def __hash__(self):
        return hash(tuple((key, _make_hashable(value)) for key, value in self._dict.items()))

    def __or__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        new = dict(self)
        new.update(other)
        return self.__class__(new)

    def __ror__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        new = dict(other)
        new.update(self)
        return new

    def __ior__(self, other):
        raise TypeError(f"'{self.__class__.__name__}' object is not mutable")

    def __getattr__(self, name: str):
        try:
            return self._dict[name]
        except KeyError as err:
            raise AttributeError(name) from err


def _make_hashable(value):
    if isinstance(value, Mapping):
        return tuple((key, _make_hashable(val)) for key, val in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_make_hashable(item) for item in value)
    return value


JsonValue = typing.Union[None, bool, int, float, str, tuple["JsonValue", ...], immutabledict]
PromptNodeDictInput = Mapping[str, typing.Any]
InputsDictInput = Mapping[str, typing.Any]
PromptDictInput = Mapping[str, typing.Any]


class InputsDict(immutabledict):
    pass


class PromptNodeDict(immutabledict):
    @property
    def class_type(self) -> str:
        return typing.cast(str, self["class_type"])

    @property
    def inputs(self) -> InputsDict:
        return typing.cast(InputsDict, self["inputs"])

    @property
    def is_changed(self):
        return self._dict.get("is_changed")


class PromptDict(immutabledict):
    def __init__(self, **kwargs):
        super().__init__(Prompt.validate(kwargs))

    @staticmethod
    def from_dict_(arg: PromptDictInput | "PromptDict", configuration=None) -> "PromptDict":
        return Prompt.validate(arg, configuration=configuration)

    def get_additional_property_(self, name: str):
        return self._dict.get(name)


def _coerce_json(value):
    if isinstance(value, decimal.Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, Mapping):
        return immutabledict({key: _coerce_json(val) for key, val in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_coerce_json(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return value


def _validate_node(node_id: str, value) -> PromptNodeDict:
    if not isinstance(value, Mapping):
        raise ApiTypeError("Prompt node must be an object", path_to_item=[node_id])
    if "class_type" not in value:
        raise ApiValueError("Prompt node is missing required property class_type", path_to_item=[node_id])
    if "inputs" not in value:
        raise ApiValueError("Prompt node is missing required property inputs", path_to_item=[node_id])
    if not isinstance(value["class_type"], str):
        raise ApiTypeError("Prompt node class_type must be a string", path_to_item=[node_id, "class_type"])
    if not isinstance(value["inputs"], Mapping):
        raise ApiTypeError("Prompt node inputs must be an object", path_to_item=[node_id, "inputs"])

    node = {key: _coerce_json(val) for key, val in value.items()}
    node["inputs"] = InputsDict(node["inputs"])
    return PromptNodeDict(node)


@dataclasses.dataclass(frozen=True)
class Prompt:
    """Compatibility wrapper for prompt validation.

    This intentionally validates only the stable prompt graph envelope. Custom
    node input values remain loose and are validated later by execution.
    """

    @classmethod
    def validate(cls, arg: PromptDictInput | PromptDict, configuration=None) -> PromptDict:
        if isinstance(arg, PromptDict):
            return arg
        if not isinstance(arg, Mapping):
            raise ApiTypeError("Prompt must be an object")

        if "nodes" in arg and "links" in arg:
            from comfy.component_model.workflow_convert import convert_ui_to_api

            arg = convert_ui_to_api(dict(arg))

        nodes = {str(node_id): _validate_node(str(node_id), value) for node_id, value in arg.items()}
        inst = object.__new__(PromptDict)
        immutabledict.__init__(inst, nodes)
        return typing.cast(PromptDict, inst)

    @staticmethod
    def from_dict_(arg: PromptDictInput | PromptDict, configuration=None) -> PromptDict:
        return Prompt.validate(arg, configuration=configuration)

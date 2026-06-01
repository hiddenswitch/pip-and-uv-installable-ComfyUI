from __future__ import annotations

import dataclasses
import typing

from comfy.api.components.schema.prompt import InputsDict, InputsDictInput, PromptNodeDict, PromptNodeDictInput


@dataclasses.dataclass(frozen=True)
class PromptNode:
    @classmethod
    def validate(cls, arg: PromptNodeDictInput | PromptNodeDict, configuration=None) -> PromptNodeDict:
        from comfy.api.components.schema.prompt import _validate_node

        return _validate_node("<node>", arg)

    @staticmethod
    def from_dict_(arg: PromptNodeDictInput | PromptNodeDict, configuration=None) -> PromptNodeDict:
        return PromptNode.validate(arg, configuration=configuration)


@dataclasses.dataclass(frozen=True)
class Inputs:
    @classmethod
    def validate(cls, arg: InputsDictInput | InputsDict, configuration=None) -> InputsDict:
        from comfy.api.components.schema.prompt import _coerce_json

        return InputsDict({key: _coerce_json(value) for key, value in typing.cast(dict, arg).items()})

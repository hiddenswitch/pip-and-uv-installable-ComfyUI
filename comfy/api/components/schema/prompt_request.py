from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping

from comfy.api.components.schema.prompt import Prompt, PromptDict, _coerce_json, immutabledict
from comfy.api.exceptions import ApiValueError

PromptRequestDictInput = Mapping[str, typing.Any]


class PromptRequestDict(immutabledict):
    @property
    def prompt(self) -> PromptDict:
        return typing.cast(PromptDict, self["prompt"])

    @property
    def client_id(self):
        return self._dict.get("client_id")

    @property
    def extra_data(self):
        return self._dict.get("extra_data")


@dataclasses.dataclass(frozen=True)
class PromptRequest:
    @classmethod
    def validate(cls, arg: PromptRequestDictInput | PromptRequestDict, configuration=None) -> PromptRequestDict:
        if isinstance(arg, PromptRequestDict):
            return arg
        if not isinstance(arg, Mapping):
            raise ApiValueError("PromptRequest must be an object")
        if "prompt" not in arg:
            raise ApiValueError("PromptRequest is missing required property prompt")
        data = {key: _coerce_json(value) for key, value in arg.items()}
        data["prompt"] = Prompt.validate(arg["prompt"])
        return PromptRequestDict(data)

    @staticmethod
    def from_dict_(arg: PromptRequestDictInput | PromptRequestDict, configuration=None) -> PromptRequestDict:
        return PromptRequest.validate(arg, configuration=configuration)

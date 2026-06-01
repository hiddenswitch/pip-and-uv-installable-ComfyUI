from __future__ import annotations

import dataclasses
import typing


class OpenApiException(Exception):
    """Base exception kept for compatibility with the old generated API package."""


def render_path(path_to_item):
    result = ""
    for pth in path_to_item:
        if isinstance(pth, int):
            result += "[{0}]".format(pth)
        else:
            result += "['{0}']".format(pth)
    return result


class ApiTypeError(OpenApiException, TypeError):
    def __init__(self, msg, path_to_item=None, valid_classes=None, key_type=None):
        self.path_to_item = path_to_item
        self.valid_classes = valid_classes
        self.key_type = key_type
        full_msg = msg
        if path_to_item:
            full_msg = "{0} at {1}".format(msg, render_path(path_to_item))
        super().__init__(full_msg)


class ApiValueError(OpenApiException, ValueError):
    def __init__(self, msg, path_to_item=None):
        self.path_to_item = path_to_item
        full_msg = msg
        if path_to_item:
            full_msg = "{0} at {1}".format(msg, render_path(path_to_item))
        super().__init__(full_msg)


class ApiAttributeError(OpenApiException, AttributeError):
    def __init__(self, msg, path_to_item=None):
        self.path_to_item = path_to_item
        full_msg = msg
        if path_to_item:
            full_msg = "{0} at {1}".format(msg, render_path(path_to_item))
        super().__init__(full_msg)


class ApiKeyError(OpenApiException, KeyError):
    def __init__(self, msg, path_to_item=None):
        self.path_to_item = path_to_item
        full_msg = msg
        if path_to_item:
            full_msg = "{0} at {1}".format(msg, render_path(path_to_item))
        super().__init__(full_msg)


T = typing.TypeVar("T")


@dataclasses.dataclass
class ApiException(OpenApiException, typing.Generic[T]):
    status: int
    reason: typing.Optional[str] = None
    api_response: typing.Optional[T] = None

    def __str__(self):
        error_message = "({0})\nReason: {1}\n".format(self.status, self.reason)
        if self.api_response:
            response = getattr(self.api_response, "response", None)
            headers = getattr(response, "headers", None)
            data = getattr(response, "data", None)
            if headers:
                error_message += "HTTP response headers: {0}\n".format(headers)
            if data:
                error_message += "HTTP response body: {0}".format(data)
        return error_message

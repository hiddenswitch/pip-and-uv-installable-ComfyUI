"""Compatibility shims for popular custom node packs.

Provides built-in implementations of commonly-used nodes from:
- ComfyUI_essentials (cubiq): SimpleMath+
- ComfyUI-KJNodes (kijai): SomethingToString
"""
import ast
import math
import operator as op

from comfy.comfy_types import IO
from comfy.node_helpers import export_custom_nodes
from comfy.nodes.package_typing import CustomNode, InputTypes


class SimpleMath(CustomNode):
    """Evaluate a math expression with variables a, b, c, d."""

    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "optional": {
                "a": (IO.ANY, {"default": 0.0}),
                "b": (IO.ANY, {"default": 0.0}),
                "c": (IO.ANY, {"default": 0.0}),
            },
            "required": {
                "value": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    CATEGORY = "essentials/utilities"
    RETURN_TYPES = ("INT", "FLOAT")
    FUNCTION = "execute"

    _OPERATORS = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.Mod: op.mod,
        ast.Eq: op.eq,
        ast.NotEq: op.ne,
        ast.Lt: op.lt,
        ast.LtE: op.le,
        ast.Gt: op.gt,
        ast.GtE: op.ge,
        ast.And: lambda x, y: x and y,
        ast.Or: lambda x, y: x or y,
        ast.Not: op.not_,
    }

    _FUNCTIONS = {
        "min": min,
        "max": max,
        "round": round,
        "sum": sum,
        "len": len,
    }

    def execute(self, value, a=0.0, b=0.0, c=0.0, d=0.0):
        if hasattr(a, "shape"):
            a = list(a.shape)
        if hasattr(b, "shape"):
            b = list(b.shape)
        if hasattr(c, "shape"):
            c = list(c.shape)
        if hasattr(d, "shape"):
            d = list(d.shape)

        if isinstance(a, str):
            a = float(a)
        if isinstance(b, str):
            b = float(b)
        if isinstance(c, str):
            c = float(c)
        if isinstance(d, str):
            d = float(d)

        variables = {"a": a, "b": b, "c": c, "d": d}
        operators = self._OPERATORS
        functions = self._FUNCTIONS

        def eval_(node):
            if isinstance(node, ast.Num):
                return node.n
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.Name):
                if node.id in variables:
                    return variables[node.id]
                return 0
            if isinstance(node, ast.BinOp):
                return operators[type(node.op)](eval_(node.left), eval_(node.right))
            if isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](eval_(node.operand))
            if isinstance(node, ast.Compare):
                left = eval_(node.left)
                for cmp_op, comparator in zip(node.ops, node.comparators):
                    if not operators[type(cmp_op)](left, eval_(comparator)):
                        return 0
                return 1
            if isinstance(node, ast.BoolOp):
                values = [eval_(v) for v in node.values]
                return operators[type(node.op)](*values)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in functions:
                    args = [eval_(arg) for arg in node.args]
                    return functions[node.func.id](*args)
            if isinstance(node, ast.Subscript):
                val = eval_(node.value)
                if isinstance(node.slice, ast.Constant):
                    return val[node.slice.value]
                return 0
            return 0

        try:
            result = eval_(ast.parse(value, mode="eval").body)
        except Exception:
            result = 0.0

        if isinstance(result, float) and math.isnan(result):
            result = 0.0

        return (round(result), float(result))


class SomethingToString(CustomNode):
    """Convert any value to a string with optional prefix/suffix."""

    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "input": (IO.ANY, {}),
            },
            "optional": {
                "prefix": ("STRING", {"default": ""}),
                "suffix": ("STRING", {"default": ""}),
            },
        }

    CATEGORY = "essentials/utilities"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    def execute(self, input, prefix="", suffix=""):
        if isinstance(input, (list, tuple)):
            val = ", ".join(str(x) for x in input)
        else:
            val = str(input)
        return (prefix + val + suffix,)


NODE_CLASS_MAPPINGS = {
    "SimpleMath+": SimpleMath,
    "SomethingToString": SomethingToString,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleMath+": "Simple Math",
    "SomethingToString": "Something To String",
}

export_custom_nodes()

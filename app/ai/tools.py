import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, ValidationError

from .config import AISettings, Component


class ToolError(ValueError):
    pass


class InvalidToolArguments(ToolError):
    pass


@dataclass(frozen=True)
class ReadOnlyFunction:
    name: str
    description: str
    arguments: type[BaseModel]
    handler: Callable[[UUID, BaseModel], object]


def strict_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()

    def check(node):
        if isinstance(node, list):
            for child in node:
                check(child)
        elif isinstance(node, dict):
            if node.get("type") == "object":
                if (
                    node.get("additionalProperties") is not False
                    or set(node.get("properties", {})) != set(node.get("required", ()))
                ):
                    raise ToolError("Structured object fields must all be required and forbid extras.")
            for child in node.values():
                check(child)

    if schema.get("type") != "object":
        raise ToolError("Structured output requires an object schema.")
    check(schema)
    return schema


class ToolRegistry:
    def __init__(self, chat_functions: Iterable[ReadOnlyFunction] = ()):
        self._functions = {}
        for function in chat_functions:
            if function.name in self._functions:
                raise ToolError("Duplicate chat function.")
            schema = strict_schema(function.arguments)
            if {"workspace_id", "workspace"} & schema["properties"].keys():
                raise ToolError("Workspace context must be supplied by the server.")
            self._functions[function.name] = function

    def request_options(self, component: Component, settings: AISettings) -> dict:
        if component == Component.RESEARCH:
            # The API's default web search return-token budget is standard.
            return {
                "tools": [{"type": "web_search", "search_context_size": settings.search_context_size}],
                "tool_choice": "required",
                "max_tool_calls": settings.max_tool_calls,
            }
        if component == Component.CHAT and self._functions:
            return {
                "tools": [
                    {
                        "type": "function", "name": function.name, "description": function.description,
                        "parameters": strict_schema(function.arguments), "strict": True,
                    }
                    for function in self._functions.values()
                ],
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "include": ["reasoning.encrypted_content"],
            }
        return {"tools": [], "tool_choice": "none"}

    def has_function(self, name: str) -> bool:
        return name in self._functions

    def dispatch(self, component: Component, workspace_id: UUID | None, name: str, arguments: str) -> str:
        if component != Component.CHAT or not isinstance(workspace_id, UUID) or name not in self._functions:
            raise ToolError("The function call is unauthorized.")
        function = self._functions[name]
        try:
            args = function.arguments.model_validate_json(arguments, strict=True)
        except ValidationError:
            raise InvalidToolArguments("The function arguments are invalid.") from None
        try:
            return json.dumps(function.handler(workspace_id, args), allow_nan=False)
        except Exception:
            raise ToolError("The authorized function could not return a result.") from None

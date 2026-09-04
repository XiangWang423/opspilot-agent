from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from .fixtures import IncidentRepository


class ToolError(Exception):
    """Base class for errors that are safe to expose to an agent."""


class UnknownToolError(ToolError):
    pass


class ToolValidationError(ToolError):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def invoke(self, arguments: dict[str, Any]) -> Any:
        _validate_arguments(self.parameters, arguments)
        return self.handler(**arguments)


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        duplicate_names = _duplicates(tool.name for tool in tools)
        if duplicate_names:
            names = ", ".join(sorted(duplicate_names))
            raise ValueError(f"Duplicate tool names: {names}")
        self._tools = {tool.name: tool for tool in tools}

    @property
    def specifications(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            tool = self._tools[name]
        except KeyError as exc:
            raise UnknownToolError(f"Unknown tool: {name}") from exc
        return tool.invoke(arguments)


def build_default_registry(repository: IncidentRepository) -> ToolRegistry:
    def search_logs(
        incident_id: str,
        service: str | None = None,
        level: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        repository.incident(incident_id)
        return [
            asdict(row)
            for row in repository.logs(
                incident_id, service=service, level=level, limit=limit
            )
        ]

    def query_metrics(
        incident_id: str,
        service: str | None = None,
        metric: str | None = None,
    ) -> list[dict[str, Any]]:
        repository.incident(incident_id)
        return [
            asdict(row)
            for row in repository.metrics(
                incident_id, service=service, metric=metric
            )
        ]

    def get_deployments(
        incident_id: str, service: str | None = None
    ) -> list[dict[str, Any]]:
        repository.incident(incident_id)
        return [
            asdict(row)
            for row in repository.deployments(incident_id, service=service)
        ]

    def search_runbooks(
        query: str, service: str | None = None
    ) -> list[dict[str, Any]]:
        return [asdict(row) for row in repository.runbooks(query=query, service=service)]

    common_incident_properties = {
        "incident_id": {
            "type": "string",
            "minLength": 1,
            "description": "Exact incident identifier, for example inc-003.",
        },
        "service": {
            "type": "string",
            "minLength": 1,
            "not": {"enum": ["all", "*"]},
            "description": (
                "Optional exact service name. Omit this field to search all services; "
                "never use 'all', '*', or an empty string."
            ),
        },
    }

    return ToolRegistry(
        [
            Tool(
                name="search_logs",
                description="Search logs captured for one reproducible incident.",
                parameters={
                    "type": "object",
                    "properties": {
                        **common_incident_properties,
                        "level": {
                            "type": "string",
                            "enum": ["DEBUG", "INFO", "WARN", "ERROR"],
                            "description": "Optional exact log severity.",
                        },
                        "limit": {"type": "integer", "minimum": 1},
                    },
                    "required": ["incident_id"],
                    "additionalProperties": False,
                },
                handler=search_logs,
            ),
            Tool(
                name="query_metrics",
                description="Query metric samples captured for one reproducible incident.",
                parameters={
                    "type": "object",
                    "properties": {
                        **common_incident_properties,
                        "metric": {
                            "type": "string",
                            "minLength": 1,
                            "not": {"enum": ["all", "*"]},
                            "description": (
                                "Optional exact metric name. Omit this field when unknown "
                                "to return all incident metrics."
                            ),
                        },
                    },
                    "required": ["incident_id"],
                    "additionalProperties": False,
                },
                handler=query_metrics,
            ),
            Tool(
                name="get_deployments",
                description="List deployments associated with one reproducible incident.",
                parameters={
                    "type": "object",
                    "properties": common_incident_properties,
                    "required": ["incident_id"],
                    "additionalProperties": False,
                },
                handler=get_deployments,
            ),
            Tool(
                name="search_runbooks",
                description="Search operational runbooks by keywords and optional service.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "minLength": 1,
                            "description": "One or more concrete diagnostic keywords.",
                        },
                        "service": common_incident_properties["service"],
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=search_runbooks,
            ),
        ]
    )


def _validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise ToolValidationError("Tool arguments must be an object")

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    missing = required - arguments.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise ToolValidationError(f"Missing required arguments: {names}")

    if schema.get("additionalProperties") is False:
        unexpected = arguments.keys() - properties.keys()
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ToolValidationError(f"Unexpected arguments: {names}")

    python_types = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    for name, value in arguments.items():
        expected_name = properties.get(name, {}).get("type")
        expected_type = python_types.get(expected_name)
        if expected_type is None:
            continue
        if expected_name in {"integer", "number"} and isinstance(value, bool):
            valid = False
        else:
            valid = isinstance(value, expected_type)
        if not valid:
            raise ToolValidationError(
                f"Argument '{name}' must be of type {expected_name}"
            )
        minimum = properties.get(name, {}).get("minimum")
        if minimum is not None and value < minimum:
            raise ToolValidationError(
                f"Argument '{name}' must be at least {minimum}"
            )

        minimum_length = properties.get(name, {}).get("minLength")
        if minimum_length is not None and len(value) < minimum_length:
            noun = "character" if minimum_length == 1 else "characters"
            raise ToolValidationError(
                f"Argument '{name}' must contain at least {minimum_length} {noun}"
            )

        allowed_values = properties.get(name, {}).get("enum")
        if allowed_values is not None and value not in allowed_values:
            choices = ", ".join(repr(item) for item in allowed_values)
            raise ToolValidationError(
                f"Argument '{name}' must be one of: {choices}"
            )

        forbidden_values = properties.get(name, {}).get("not", {}).get("enum")
        if forbidden_values is not None and value in forbidden_values:
            choices = ", ".join(repr(item) for item in forbidden_values)
            raise ToolValidationError(
                f"Argument '{name}' must not be one of: {choices}"
            )


def _duplicates(names: list[str] | Any) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    return duplicates

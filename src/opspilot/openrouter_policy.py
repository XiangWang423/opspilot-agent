from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .agent import AgentState, FinalDiagnosis, ToolCall


class ModelResponseError(RuntimeError):
    """The provider returned a response the policy cannot safely interpret."""


class ChatClient(Protocol):
    def complete(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class OpenRouterChatClient:
    """Small standard-library client for OpenRouter chat completions."""

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout_seconds: float = 30,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc

        if not isinstance(result, dict):
            raise ModelResponseError("OpenRouter response must be a JSON object")
        return result


class OpenRouterPolicy:
    """Translate OpenRouter chat responses into runner decisions."""

    _SYSTEM_PROMPT = (
        "You diagnose incidents using only the supplied tools and observations. "
        "Request one tool at a time. Omit optional filters when their exact values "
        "are unknown; never invent placeholders such as 'all', '*', or an empty "
        "string. Start broad, then narrow using values found in observations. "
        "Do not repeat a narrower query when the broader result already contains "
        "the needed evidence. Prefer independent evidence sources. "
        "When enough evidence exists, return only a "
        "JSON object with root_cause (string) and confidence (number from 0 to 1)."
    )

    def __init__(self, client: ChatClient, model: str) -> None:
        if not model:
            raise ValueError("model must not be empty")
        self.client = client
        self.model = model

    def decide(
        self, state: AgentState, tool_specs: list[dict[str, Any]]
    ) -> ToolCall | FinalDiagnosis:
        payload = {
            "model": self.model,
            "messages": self._messages(state),
            "tools": [
                {"type": "function", "function": specification}
                for specification in tool_specs
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": 0,
        }
        response = self.client.complete(payload)
        message = self._message(response)
        tool_calls = message.get("tool_calls") or []

        if tool_calls:
            if len(tool_calls) != 1:
                raise ModelResponseError("Expected at most one tool call")
            return self._tool_call(tool_calls[0])

        return self._final_diagnosis(message.get("content"))

    def _messages(self, state: AgentState) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._task_message(state),
            },
        ]

        for index, observation in enumerate(state.observations, start=1):
            call_id = observation.call_id or f"local_call_{index}"
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": observation.tool_name,
                                "arguments": json.dumps(observation.arguments),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        {"output": observation.output, "error": observation.error},
                        default=str,
                    ),
                }
            )

        return messages

    @staticmethod
    def _task_message(state: AgentState) -> str:
        task = f"Investigate incident {state.incident_id}."
        if state.max_steps is None or state.steps_remaining is None:
            return task
        return (
            f"{task} This is decision step {state.steps} of {state.max_steps}; "
            f"{state.steps_remaining} decision(s) remain including this one. "
            "Reserve a decision for the final diagnosis and stop gathering evidence "
            "once the root cause is supported."
        )

    @staticmethod
    def _message(response: dict[str, Any]) -> dict[str, Any]:
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelResponseError("Response has no assistant message") from exc
        if not isinstance(message, dict):
            raise ModelResponseError("Assistant message must be an object")
        return message

    @staticmethod
    def _tool_call(raw_call: Any) -> ToolCall:
        try:
            call_id = raw_call["id"]
            function = raw_call["function"]
            name = function["name"]
            raw_arguments = function["arguments"]
            arguments = json.loads(raw_arguments)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ModelResponseError("Malformed model tool call") from exc

        if not isinstance(call_id, str) or not isinstance(name, str):
            raise ModelResponseError("Tool call id and name must be strings")
        if not isinstance(arguments, dict):
            raise ModelResponseError("Tool call arguments must be a JSON object")
        return ToolCall(name, arguments, call_id=call_id)

    @staticmethod
    def _final_diagnosis(content: Any) -> FinalDiagnosis:
        if not isinstance(content, str):
            raise ModelResponseError("Final diagnosis content must be text")
        try:
            parsed = json.loads(content)
            root_cause = parsed["root_cause"]
            confidence = parsed["confidence"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ModelResponseError("Final diagnosis must be valid JSON") from exc

        if not isinstance(root_cause, str) or not root_cause.strip():
            raise ModelResponseError("root_cause must be a non-empty string")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ModelResponseError("confidence must be a number")
        if not 0 <= confidence <= 1:
            raise ModelResponseError("confidence must be between 0 and 1")
        return FinalDiagnosis(root_cause=root_cause, confidence=float(confidence))

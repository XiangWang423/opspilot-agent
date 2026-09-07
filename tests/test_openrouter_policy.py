import json
from pathlib import Path
import unittest

from opspilot.agent import AgentRunner, AgentState, FinalDiagnosis, ToolCall
from opspilot.fixtures import IncidentRepository
from opspilot.openrouter_policy import ModelResponseError, OpenRouterPolicy
from opspilot.tools import build_default_registry


FIXTURE_PATH = Path(__file__).parents[1] / "data" / "incidents.json"


class FakeChatClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    def complete(self, payload: dict[str, object]) -> dict[str, object]:
        self.requests.append(payload)
        return self.response


class ScriptedChatClient:
    def __init__(self, *responses: dict[str, object]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def complete(self, payload: dict[str, object]) -> dict[str, object]:
        self.requests.append(payload)
        return self.responses.pop(0)


class OpenRouterPolicyTests(unittest.TestCase):
    def test_converts_model_tool_call_to_policy_tool_call(self) -> None:
        client = FakeChatClient(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_logs_001",
                                    "type": "function",
                                    "function": {
                                        "name": "search_logs",
                                        "arguments": json.dumps(
                                            {
                                                "incident_id": "inc-003",
                                                "level": "ERROR",
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )
        policy = OpenRouterPolicy(client=client, model="test/model")

        decision = policy.decide(
            AgentState(incident_id="inc-003"),
            [
                {
                    "name": "search_logs",
                    "description": "Search incident logs.",
                    "parameters": {
                        "type": "object",
                        "properties": {"incident_id": {"type": "string"}},
                        "required": ["incident_id"],
                    },
                }
            ],
        )

        self.assertEqual(
            decision,
            ToolCall(
                "search_logs",
                {"incident_id": "inc-003", "level": "ERROR"},
                call_id="call_logs_001",
            ),
        )
        self.assertEqual(client.requests[0]["model"], "test/model")
        self.assertIn(
            "Omit optional filters",
            client.requests[0]["messages"][0]["content"],
        )
        self.assertEqual(
            client.requests[0]["tools"][0]["function"]["name"], "search_logs"
        )

    def test_tells_model_the_remaining_decision_budget(self) -> None:
        client = FakeChatClient(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"root_cause": "configuration error", "confidence": 0.8}
                            ),
                        }
                    }
                ]
            }
        )
        policy = OpenRouterPolicy(client=client, model="test/model")
        state = AgentState(incident_id="inc-003", steps=3, max_steps=4)

        policy.decide(state, [])

        task = client.requests[0]["messages"][1]["content"]
        self.assertIn("decision step 3 of 4", task)
        self.assertIn("2 decision(s) remain", task)
        self.assertIn("Reserve a decision for the final diagnosis", task)

    def test_includes_incident_context_without_ground_truth(self) -> None:
        client = FakeChatClient(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"root_cause": "configuration error", "confidence": 0.8}
                            ),
                        }
                    }
                ]
            }
        )
        policy = OpenRouterPolicy(client=client, model="test/model")
        state = AgentState(
            incident_id="inc-003",
            context={
                "service": "auth-api",
                "severity": "SEV-1",
                "alert": "Login failures rose after deployment.",
            },
        )

        policy.decide(state, [])

        task = client.requests[0]["messages"][1]["content"]
        self.assertIn("auth-api", task)
        self.assertIn("Login failures", task)
        self.assertNotIn("ground_truth", task)

    def test_converts_json_text_to_final_diagnosis(self) -> None:
        client = FakeChatClient(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "root_cause": "production used the staging issuer",
                                    "confidence": 0.96,
                                }
                            ),
                        }
                    }
                ]
            }
        )
        policy = OpenRouterPolicy(client=client, model="test/model")

        decision = policy.decide(AgentState(incident_id="inc-003"), [])

        self.assertEqual(
            decision,
            FinalDiagnosis(
                root_cause="production used the staging issuer",
                confidence=0.96,
            ),
        )

    def test_runner_returns_tool_result_to_model_before_diagnosis(self) -> None:
        client = ScriptedChatClient(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_logs_002",
                                    "type": "function",
                                    "function": {
                                        "name": "search_logs",
                                        "arguments": json.dumps(
                                            {
                                                "incident_id": "inc-003",
                                                "service": "auth-api",
                                                "level": "ERROR",
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "root_cause": (
                                        "auth-api used the staging JWT issuer "
                                        "in production"
                                    ),
                                    "confidence": 0.97,
                                }
                            ),
                        }
                    }
                ]
            },
        )
        repository = IncidentRepository(FIXTURE_PATH)
        registry = build_default_registry(repository)
        runner = AgentRunner(
            OpenRouterPolicy(client=client, model="test/model"),
            registry,
            max_steps=3,
        )

        result = runner.run("inc-003")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(result.observations), 1)
        self.assertEqual(result.observations[0].call_id, "call_logs_002")
        self.assertIn("invalid issuer", result.observations[0].output[0]["message"])

        second_messages = client.requests[1]["messages"]
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertEqual(second_messages[-1]["tool_call_id"], "call_logs_002")
        self.assertIn("invalid issuer", second_messages[-1]["content"])
        second_tool_names = {
            item["function"]["name"] for item in client.requests[1]["tools"]
        }
        self.assertNotIn("search_logs", second_tool_names)

    def test_reserves_last_decision_for_final_diagnosis(self) -> None:
        client = FakeChatClient(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "root_cause": "production used the staging issuer",
                                    "confidence": 0.95,
                                }
                            ),
                        }
                    }
                ]
            }
        )
        policy = OpenRouterPolicy(client=client, model="test/model")
        state = AgentState(incident_id="inc-003", steps=4, max_steps=4)

        decision = policy.decide(state, [])

        self.assertIsInstance(decision, FinalDiagnosis)
        self.assertEqual(client.requests[0]["tool_choice"], "none")

    def test_rejects_tool_call_with_invalid_json_arguments(self) -> None:
        client = FakeChatClient(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_bad_001",
                                    "type": "function",
                                    "function": {
                                        "name": "search_logs",
                                        "arguments": "{not valid json",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )
        policy = OpenRouterPolicy(client=client, model="test/model")

        with self.assertRaisesRegex(ModelResponseError, "Malformed model tool call"):
            policy.decide(AgentState(incident_id="inc-003"), [])

    def test_rejects_final_diagnosis_with_invalid_confidence(self) -> None:
        client = FakeChatClient(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "root_cause": "unknown configuration problem",
                                    "confidence": 1.7,
                                }
                            ),
                        }
                    }
                ]
            }
        )
        policy = OpenRouterPolicy(client=client, model="test/model")

        with self.assertRaisesRegex(
            ModelResponseError, "confidence must be between 0 and 1"
        ):
            policy.decide(AgentState(incident_id="inc-003"), [])


if __name__ == "__main__":
    unittest.main()

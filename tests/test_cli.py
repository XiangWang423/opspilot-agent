from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from opspilot.cli import main, result_payload, run_diagnosis


FIXTURE_PATH = Path(__file__).parents[1] / "data" / "incidents.json"


class ScriptedChatClient:
    def __init__(self, *responses: dict[str, object]) -> None:
        self.responses = list(responses)

    def complete(self, payload: dict[str, object]) -> dict[str, object]:
        return self.responses.pop(0)


class CliTests(unittest.TestCase):
    def test_main_rejects_missing_api_key_without_network(self) -> None:
        stderr = io.StringIO()

        with patch.dict(os.environ, {}, clear=True), redirect_stderr(stderr):
            exit_code = main(["inc-003"])

        self.assertEqual(exit_code, 2)
        self.assertIn("OPENROUTER_API_KEY is not set", stderr.getvalue())

    def test_runs_complete_diagnosis_without_network(self) -> None:
        client = ScriptedChatClient(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_deploy_001",
                                    "type": "function",
                                    "function": {
                                        "name": "get_deployments",
                                        "arguments": json.dumps(
                                            {
                                                "incident_id": "inc-003",
                                                "service": "auth-api",
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
                                    "root_cause": "staging issuer reached production",
                                    "confidence": 0.94,
                                }
                            ),
                        }
                    }
                ]
            },
        )

        result = run_diagnosis(
            incident_id="inc-003",
            data_path=FIXTURE_PATH,
            model="test/model",
            client=client,
            max_steps=3,
        )
        payload = result_payload("inc-003", result)

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["tool_calls"], 1)
        self.assertEqual(payload["diagnosis"]["confidence"], 0.94)

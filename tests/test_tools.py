from pathlib import Path
import unittest

from opspilot.fixtures import IncidentRepository
from opspilot.tools import (
    ToolValidationError,
    UnknownToolError,
    build_default_registry,
)


FIXTURE_PATH = Path(__file__).parents[1] / "data" / "incidents.json"


class ToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        repository = IncidentRepository(FIXTURE_PATH)
        self.registry = build_default_registry(repository)

    def test_exposes_four_tool_contracts(self) -> None:
        names = {tool["name"] for tool in self.registry.specifications}

        self.assertEqual(
            names,
            {
                "search_logs",
                "query_metrics",
                "get_deployments",
                "search_runbooks",
            },
        )

    def test_invokes_read_only_log_tool(self) -> None:
        result = self.registry.invoke(
            "search_logs",
            {"incident_id": "inc-003", "service": "auth-api", "level": "ERROR"},
        )

        self.assertEqual(len(result), 1)
        self.assertIn("invalid issuer", result[0]["message"])

    def test_rejects_missing_required_argument_before_handler(self) -> None:
        with self.assertRaisesRegex(
            ToolValidationError, "Missing required arguments: incident_id"
        ):
            self.registry.invoke("query_metrics", {"service": "checkout-api"})

    def test_rejects_unexpected_argument_before_handler(self) -> None:
        with self.assertRaisesRegex(
            ToolValidationError, "Unexpected arguments: typo"
        ):
            self.registry.invoke(
                "get_deployments", {"incident_id": "inc-001", "typo": True}
            )

    def test_rejects_argument_with_wrong_type(self) -> None:
        with self.assertRaisesRegex(
            ToolValidationError, "Argument 'limit' must be of type integer"
        ):
            self.registry.invoke(
                "search_logs", {"incident_id": "inc-001", "limit": "ten"}
            )

    def test_rejects_limit_below_minimum(self) -> None:
        with self.assertRaisesRegex(
            ToolValidationError, "Argument 'limit' must be at least 1"
        ):
            self.registry.invoke(
                "search_logs",
                {"incident_id": "inc-001", "limit": -1},
            )

    def test_rejects_unknown_tool(self) -> None:
        with self.assertRaisesRegex(UnknownToolError, "Unknown tool: restart_service"):
            self.registry.invoke("restart_service", {"service": "checkout-api"})


if __name__ == "__main__":
    unittest.main()


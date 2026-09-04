from pathlib import Path
import unittest

from opspilot.agent import AgentRunner, AgentState, FinalDiagnosis, ToolCall
from opspilot.fixtures import IncidentRepository
from opspilot.tools import build_default_registry


FIXTURE_PATH = Path(__file__).parents[1] / "data" / "incidents.json"


class AuthenticationIncidentPolicy:
    """Deterministic policy used to test orchestration without an LLM."""

    def decide(
        self, state: AgentState, tool_specs: list[dict[str, object]]
    ) -> ToolCall | FinalDiagnosis:
        if not state.observations:
            return ToolCall(
                "search_logs",
                {
                    "incident_id": state.incident_id,
                    "service": "auth-api",
                    "level": "ERROR",
                },
            )

        if len(state.observations) == 1:
            return ToolCall(
                "get_deployments",
                {"incident_id": state.incident_id, "service": "auth-api"},
            )

        return FinalDiagnosis(
            root_cause=(
                "auth-api version 3.4.2 deployed the staging JWT issuer "
                "configuration to production"
            ),
            confidence=0.98,
        )


class RecoveringAuthenticationPolicy:
    """First emits invalid arguments, then repairs them from the observation."""

    def decide(
        self, state: AgentState, tool_specs: list[dict[str, object]]
    ) -> ToolCall | FinalDiagnosis:
        if not state.observations:
            return ToolCall("search_logs", {"service": "auth-api"})

        if state.observations[-1].error is not None:
            return ToolCall(
                "search_logs",
                {
                    "incident_id": state.incident_id,
                    "service": "auth-api",
                    "level": "ERROR",
                },
            )

        return FinalDiagnosis(
            root_cause="auth-api used the staging JWT issuer in production",
            confidence=0.95,
        )


class NeverEndingPolicy:
    """A faulty policy that keeps requesting the same tool forever."""

    def decide(
        self, state: AgentState, tool_specs: list[dict[str, object]]
    ) -> ToolCall | FinalDiagnosis:
        return ToolCall(
            "search_logs",
            {"incident_id": state.incident_id, "service": "auth-api"},
        )


class AgentRunnerTests(unittest.TestCase):
    def test_collects_evidence_before_returning_a_diagnosis(self) -> None:
        repository = IncidentRepository(FIXTURE_PATH)
        registry = build_default_registry(repository)
        runner = AgentRunner(AuthenticationIncidentPolicy(), registry, max_steps=4)

        result = runner.run("inc-003")

        self.assertEqual(result.status, "completed")
        self.assertIsNotNone(result.diagnosis)
        self.assertIn("staging JWT issuer", result.diagnosis.root_cause)
        self.assertEqual(
            [observation.tool_name for observation in result.observations],
            ["search_logs", "get_deployments"],
        )
        self.assertEqual(result.steps, 3)

    def test_recovers_after_invalid_tool_arguments(self) -> None:
        repository = IncidentRepository(FIXTURE_PATH)
        registry = build_default_registry(repository)
        runner = AgentRunner(RecoveringAuthenticationPolicy(), registry, max_steps=4)

        result = runner.run("inc-003")

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.observations), 2)
        self.assertIn("Missing required arguments: incident_id", result.observations[0].error)
        self.assertIsNone(result.observations[1].error)
        self.assertIn("JWT validation failed", result.observations[1].output[0]["message"])
        self.assertEqual(result.steps, 3)

    def test_stops_a_policy_that_never_finishes(self) -> None:
        repository = IncidentRepository(FIXTURE_PATH)
        registry = build_default_registry(repository)
        runner = AgentRunner(NeverEndingPolicy(), registry, max_steps=2)

        result = runner.run("inc-003")

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(result.observations), 2)
        self.assertEqual(result.error, "Maximum step count (2) reached")

    def test_detects_repeated_identical_tool_calls(self) -> None:
        repository = IncidentRepository(FIXTURE_PATH)
        registry = build_default_registry(repository)
        runner = AgentRunner(
            NeverEndingPolicy(),
            registry,
            max_steps=6,
            max_identical_tool_calls=2,
        )

        result = runner.run("inc-003")

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.steps, 3)
        self.assertEqual(len(result.observations), 2)
        self.assertIn("Repeated tool call detected", result.error)


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import unittest

from opspilot.agent import AgentResult, FinalDiagnosis, Observation
from opspilot.evaluation import evaluate, report_payload
from opspilot.fixtures import IncidentRepository


FIXTURE_PATH = Path(__file__).parents[1] / "data" / "incidents.json"


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = IncidentRepository(FIXTURE_PATH)

    def test_scores_a_correct_grounded_diagnosis(self) -> None:
        def diagnose(incident_id: str) -> AgentResult:
            self.assertEqual(incident_id, "inc-003")
            return AgentResult(
                status="completed",
                diagnosis=FinalDiagnosis(
                    "Version 3.4.2 put the staging JWT issuer in production.", 0.96
                ),
                observations=(
                    Observation(
                        "search_logs",
                        {"incident_id": incident_id},
                        output=[
                            {"evidence_id": "log-auth-invalid-issuer"}
                        ],
                    ),
                    Observation(
                        "query_metrics",
                        {"incident_id": incident_id},
                        output=[
                            {"evidence_id": "metric-auth-login-failure"}
                        ],
                    ),
                    Observation(
                        "get_deployments",
                        {"incident_id": incident_id},
                        output=[{"deployment_id": "deploy-auth-342"}],
                    ),
                ),
                steps=4,
            )

        report = evaluate(self.repository, diagnose, incident_ids=["inc-003"])
        case = report.cases[0]

        self.assertTrue(case.correct)
        self.assertTrue(case.grounded)
        self.assertIn("staging JWT issuer", case.predicted_root_cause)
        self.assertEqual(case.confidence, 0.96)
        self.assertEqual(case.diagnosis_term_recall, 1.0)
        self.assertEqual(case.evidence_recall, 1.0)
        self.assertEqual(report.grounded_accuracy, 1.0)
        self.assertEqual(report_payload(report)["summary"]["cases"], 1)

    def test_report_rounds_display_metrics(self) -> None:
        result = AgentResult(
            status="failed",
            diagnosis=None,
            observations=(),
            steps=1,
            error="failed",
        )
        report = evaluate(
            self.repository,
            lambda _: result,
            incident_ids=["inc-001", "inc-002", "inc-003"],
        )

        payload = report_payload(report)

        self.assertEqual(payload["summary"]["average_steps"], 1.0)
        self.assertEqual(payload["cases"][0]["diagnosis_term_recall"], 0.0)

    def test_separates_correctness_from_evidence_grounding(self) -> None:
        result = AgentResult(
            status="completed",
            diagnosis=FinalDiagnosis(
                "fraud-api latency made payment-api requests timeout", 0.9
            ),
            observations=(
                Observation(
                    "search_logs",
                    {"incident_id": "inc-002"},
                    output=[{"evidence_id": "log-payment-fraud-timeout"}],
                ),
            ),
            steps=2,
        )

        report = evaluate(
            self.repository, lambda _: result, incident_ids=["inc-002"]
        )
        case = report.cases[0]

        self.assertTrue(case.correct)
        self.assertFalse(case.grounded)
        self.assertEqual(case.evidence_recall, 0.5)
        self.assertEqual(
            case.missing_evidence_ids, ("metric-fraud-latency",)
        )

    def test_matches_compound_term_written_with_a_space(self) -> None:
        result = AgentResult(
            status="completed",
            diagnosis=FinalDiagnosis(
                "fraud-api latency made payment-api authorization time out", 0.9
            ),
            observations=(),
            steps=1,
        )

        report = evaluate(
            self.repository, lambda _: result, incident_ids=["inc-002"]
        )

        self.assertIn("timeout", report.cases[0].matched_terms)
        self.assertTrue(report.cases[0].correct)

    def test_records_one_case_failure_without_stopping_the_suite(self) -> None:
        def diagnose(incident_id: str) -> AgentResult:
            if incident_id == "inc-001":
                raise RuntimeError("provider unavailable")
            return AgentResult(
                status="failed",
                diagnosis=None,
                observations=(),
                steps=3,
                error="step budget reached",
            )

        report = evaluate(
            self.repository,
            diagnose,
            incident_ids=["inc-001", "inc-002"],
        )

        self.assertEqual(len(report.cases), 2)
        self.assertIn("provider unavailable", report.cases[0].error)
        self.assertEqual(report.completion_rate, 0.0)

    def test_rejects_invalid_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            evaluate(self.repository, lambda _: None, minimum_term_recall=1.1)


if __name__ == "__main__":
    unittest.main()

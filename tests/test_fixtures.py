from pathlib import Path
import unittest

from opspilot.fixtures import IncidentRepository


FIXTURE_PATH = Path(__file__).parents[1] / "data" / "incidents.json"


class IncidentRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = IncidentRepository(FIXTURE_PATH)

    def test_loads_known_incident_and_ground_truth(self) -> None:
        incident = self.repository.incident("inc-001")

        self.assertEqual(incident.service, "checkout-api")
        self.assertEqual(
            incident.ground_truth["category"], "resource_exhaustion"
        )

    def test_log_filters_are_deterministic(self) -> None:
        logs = self.repository.logs(
            "inc-001", service="checkout-api", level="ERROR"
        )

        self.assertEqual(len(logs), 1)
        self.assertIn("pool exhausted", logs[0].message)

    def test_metric_samples_are_chronological(self) -> None:
        samples = self.repository.metrics(
            "inc-001", service="checkout-api", metric="db_pool_in_use"
        )

        self.assertEqual([sample.value for sample in samples], [18, 20])

    def test_runbook_search_uses_keywords(self) -> None:
        runbooks = self.repository.runbooks(
            query="connection pool timeout", service="checkout-api"
        )

        self.assertEqual(runbooks[0].runbook_id, "rb-db-pool")


if __name__ == "__main__":
    unittest.main()


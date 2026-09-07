from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, TypeVar

from .domain import Deployment, Incident, LogEntry, MetricSample, Runbook


T = TypeVar("T")


class FixtureFormatError(ValueError):
    """Raised when a fixture file is missing required top-level collections."""


class IncidentRepository:
    """Read-only access to deterministic incident evidence."""

    REQUIRED_COLLECTIONS = {
        "incidents",
        "logs",
        "metrics",
        "deployments",
        "runbooks",
    }

    def __init__(self, fixture_path: str | Path):
        self.fixture_path = Path(fixture_path)
        payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        missing = self.REQUIRED_COLLECTIONS - payload.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise FixtureFormatError(f"Fixture is missing collections: {names}")

        self._incidents = tuple(Incident(**item) for item in payload["incidents"])
        self._logs = tuple(LogEntry(**item) for item in payload["logs"])
        self._metrics = tuple(MetricSample(**item) for item in payload["metrics"])
        self._deployments = tuple(
            Deployment(**item) for item in payload["deployments"]
        )
        self._runbooks = tuple(
            Runbook(tags=tuple(item["tags"]), **{k: v for k, v in item.items() if k != "tags"})
            for item in payload["runbooks"]
        )

    def incident(self, incident_id: str) -> Incident:
        for incident in self._incidents:
            if incident.incident_id == incident_id:
                return incident
        raise KeyError(f"Unknown incident: {incident_id}")

    @property
    def incident_ids(self) -> tuple[str, ...]:
        """Stable fixture order for repeatable evaluation runs."""

        return tuple(incident.incident_id for incident in self._incidents)

    def logs(
        self,
        incident_id: str,
        *,
        service: str | None = None,
        level: str | None = None,
        limit: int = 20,
    ) -> list[LogEntry]:
        rows = self._for_incident(self._logs, incident_id)
        if service:
            rows = [row for row in rows if row.service == service]
        if level:
            rows = [row for row in rows if row.level.lower() == level.lower()]
        return sorted(rows, key=lambda row: row.timestamp, reverse=True)[:limit]

    def metrics(
        self,
        incident_id: str,
        *,
        service: str | None = None,
        metric: str | None = None,
    ) -> list[MetricSample]:
        rows = self._for_incident(self._metrics, incident_id)
        if service:
            rows = [row for row in rows if row.service == service]
        if metric:
            rows = [row for row in rows if row.metric == metric]
        return sorted(rows, key=lambda row: row.timestamp)

    def deployments(
        self, incident_id: str, *, service: str | None = None
    ) -> list[Deployment]:
        rows = self._for_incident(self._deployments, incident_id)
        if service:
            rows = [row for row in rows if row.service == service]
        return sorted(rows, key=lambda row: row.deployed_at, reverse=True)

    def runbooks(self, *, query: str, service: str | None = None) -> list[Runbook]:
        query_tokens = _tokens(query)
        candidates = self._runbooks
        if service:
            candidates = tuple(row for row in candidates if row.service == service)

        scored: list[tuple[int, Runbook]] = []
        for runbook in candidates:
            searchable = " ".join(
                [runbook.title, runbook.content, " ".join(runbook.tags)]
            )
            score = len(query_tokens & _tokens(searchable))
            if score:
                scored.append((score, runbook))
        scored.sort(key=lambda item: (-item[0], item[1].runbook_id))
        return [runbook for _, runbook in scored]

    @staticmethod
    def _for_incident(rows: Iterable[T], incident_id: str) -> list[T]:
        return [row for row in rows if getattr(row, "incident_id") == incident_id]


def _tokens(text: str) -> set[str]:
    normalized = "".join(character.lower() if character.isalnum() else " " for character in text)
    return {token for token in normalized.split() if token}

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Incident:
    incident_id: str
    service: str
    started_at: str
    alert: str
    severity: str
    ground_truth: dict[str, Any]


@dataclass(frozen=True)
class LogEntry:
    incident_id: str
    timestamp: str
    service: str
    level: str
    message: str
    trace_id: str | None = None


@dataclass(frozen=True)
class MetricSample:
    incident_id: str
    timestamp: str
    service: str
    metric: str
    value: float
    unit: str


@dataclass(frozen=True)
class Deployment:
    incident_id: str
    deployment_id: str
    service: str
    version: str
    deployed_at: str
    commit_sha: str
    summary: str


@dataclass(frozen=True)
class Runbook:
    runbook_id: str
    title: str
    service: str
    tags: tuple[str, ...]
    content: str


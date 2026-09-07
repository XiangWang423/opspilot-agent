from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Callable, Iterable, Sequence

from .agent import AgentResult
from .cli import run_diagnosis
from .fixtures import IncidentRepository
from .openrouter_policy import OpenRouterChatClient


Diagnose = Callable[[str], AgentResult]


@dataclass(frozen=True)
class CaseEvaluation:
    incident_id: str
    completed: bool
    correct: bool
    grounded: bool
    diagnosis_term_recall: float
    evidence_recall: float
    matched_terms: tuple[str, ...]
    missing_evidence_ids: tuple[str, ...]
    tool_calls: int
    steps: int
    error: str | None = None


@dataclass(frozen=True)
class EvaluationReport:
    cases: tuple[CaseEvaluation, ...]
    completion_rate: float
    diagnosis_accuracy: float
    grounded_accuracy: float
    average_evidence_recall: float
    average_tool_calls: float
    average_steps: float


def evaluate(
    repository: IncidentRepository,
    diagnose: Diagnose,
    *,
    incident_ids: Sequence[str] | None = None,
    minimum_term_recall: float = 0.6,
) -> EvaluationReport:
    """Evaluate diagnoses against fixture truth without an LLM-as-judge."""

    if not 0 <= minimum_term_recall <= 1:
        raise ValueError("minimum_term_recall must be between 0 and 1")
    selected_ids = tuple(incident_ids or repository.incident_ids)
    cases = tuple(
        _evaluate_case(repository, incident_id, diagnose, minimum_term_recall)
        for incident_id in selected_ids
    )
    count = len(cases)
    if not count:
        return EvaluationReport((), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return EvaluationReport(
        cases=cases,
        completion_rate=_mean(case.completed for case in cases),
        diagnosis_accuracy=_mean(case.correct for case in cases),
        grounded_accuracy=_mean(case.grounded for case in cases),
        average_evidence_recall=_mean(case.evidence_recall for case in cases),
        average_tool_calls=_mean(case.tool_calls for case in cases),
        average_steps=_mean(case.steps for case in cases),
    )


def _evaluate_case(
    repository: IncidentRepository,
    incident_id: str,
    diagnose: Diagnose,
    minimum_term_recall: float,
) -> CaseEvaluation:
    incident = repository.incident(incident_id)
    expected_terms = tuple(incident.ground_truth.get("diagnosis_terms", ()))
    expected_evidence = set(incident.ground_truth.get("supporting_evidence", ()))

    try:
        result = diagnose(incident_id)
    except Exception as exc:  # Keep a multi-case evaluation running.
        return CaseEvaluation(
            incident_id=incident_id,
            completed=False,
            correct=False,
            grounded=False,
            diagnosis_term_recall=0.0,
            evidence_recall=0.0,
            matched_terms=(),
            missing_evidence_ids=tuple(sorted(expected_evidence)),
            tool_calls=0,
            steps=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    diagnosis_text = result.diagnosis.root_cause if result.diagnosis else ""
    normalized_diagnosis = _normalize(diagnosis_text)
    matched_terms = tuple(
        term for term in expected_terms if _contains_term(normalized_diagnosis, term)
    )
    term_recall = len(matched_terms) / len(expected_terms) if expected_terms else 0.0

    observed_evidence = _observed_evidence_ids(result)
    found_evidence = expected_evidence & observed_evidence
    evidence_recall = (
        len(found_evidence) / len(expected_evidence) if expected_evidence else 1.0
    )
    completed = result.status == "completed" and result.diagnosis is not None
    correct = completed and term_recall >= minimum_term_recall
    grounded = correct and evidence_recall == 1.0
    return CaseEvaluation(
        incident_id=incident_id,
        completed=completed,
        correct=correct,
        grounded=grounded,
        diagnosis_term_recall=term_recall,
        evidence_recall=evidence_recall,
        matched_terms=matched_terms,
        missing_evidence_ids=tuple(sorted(expected_evidence - observed_evidence)),
        tool_calls=len(result.observations),
        steps=result.steps,
        error=result.error,
    )


def _observed_evidence_ids(result: AgentResult) -> set[str]:
    identifiers: set[str] = set()
    for observation in result.observations:
        rows = observation.output if isinstance(observation.output, list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in ("evidence_id", "deployment_id", "runbook_id"):
                value = row.get(key)
                if isinstance(value, str):
                    identifiers.add(value)
    return identifiers


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _contains_term(normalized_diagnosis: str, term: str) -> bool:
    normalized_term = _normalize(term)
    if f" {normalized_term} " in f" {normalized_diagnosis} ":
        return True
    compact_term = normalized_term.replace(" ", "")
    compact_diagnosis = normalized_diagnosis.replace(" ", "")
    return len(compact_term) >= 6 and compact_term in compact_diagnosis


def _mean(values: Iterable[float | int | bool]) -> float:
    collected = tuple(float(value) for value in values)
    return sum(collected) / len(collected)


def report_payload(report: EvaluationReport) -> dict[str, object]:
    return {
        "summary": {
            "cases": len(report.cases),
            "completion_rate": report.completion_rate,
            "diagnosis_accuracy": report.diagnosis_accuracy,
            "grounded_accuracy": report.grounded_accuracy,
            "average_evidence_recall": report.average_evidence_recall,
            "average_tool_calls": report.average_tool_calls,
            "average_steps": report.average_steps,
        },
        "cases": [asdict(case) for case in report.cases],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate OpsPilot on deterministic incident fixtures."
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/incidents.json")
    )
    parser.add_argument("--model", default="openai/gpt-5-nano")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument(
        "--incident", action="append", dest="incident_ids",
        help="Evaluate one incident; repeat to select multiple incidents.",
    )
    parser.add_argument("--minimum-term-recall", type=float, default=0.6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2

    repository = IncidentRepository(args.data)
    client = OpenRouterChatClient(api_key)

    def diagnose(incident_id: str) -> AgentResult:
        return run_diagnosis(
            incident_id=incident_id,
            data_path=args.data,
            model=args.model,
            client=client,
            max_steps=args.max_steps,
        )

    report = evaluate(
        repository,
        diagnose,
        incident_ids=args.incident_ids,
        minimum_term_recall=args.minimum_term_recall,
    )
    print(json.dumps(report_payload(report), indent=2))
    return 0 if report.grounded_accuracy == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

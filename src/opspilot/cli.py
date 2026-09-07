from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from .agent import AgentResult, AgentRunner
from .fixtures import IncidentRepository
from .openrouter_policy import ChatClient, OpenRouterChatClient, OpenRouterPolicy
from .tools import build_default_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose a reproducible incident with OpsPilot."
    )
    parser.add_argument("incident_id", help="Incident identifier, for example inc-003")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/incidents.json"),
        help="Path to the deterministic incident fixture file",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-5-nano",
        help="OpenRouter model slug",
    )
    parser.add_argument("--max-steps", type=int, default=6)
    return parser


def run_diagnosis(
    *,
    incident_id: str,
    data_path: Path,
    model: str,
    client: ChatClient,
    max_steps: int,
) -> AgentResult:
    repository = IncidentRepository(data_path)
    incident = repository.incident(incident_id)
    tools = build_default_registry(repository)
    policy = OpenRouterPolicy(client=client, model=model)
    context = {
        "service": incident.service,
        "started_at": incident.started_at,
        "alert": incident.alert,
        "severity": incident.severity,
    }
    return AgentRunner(policy, tools, max_steps=max_steps).run(
        incident_id, context=context
    )


def result_payload(incident_id: str, result: AgentResult) -> dict[str, object]:
    return {
        "incident_id": incident_id,
        "status": result.status,
        "diagnosis": asdict(result.diagnosis) if result.diagnosis else None,
        "steps": result.steps,
        "tool_calls": len(result.observations),
        "observations": [asdict(item) for item in result.observations],
        "error": result.error,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2

    try:
        result = run_diagnosis(
            incident_id=args.incident_id,
            data_path=args.data,
            model=args.model,
            client=OpenRouterChatClient(api_key),
            max_steps=args.max_steps,
        )
    except RuntimeError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 2

    print(json.dumps(result_payload(args.incident_id, result), indent=2))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

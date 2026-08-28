# OpsPilot Agent

OpsPilot is a portfolio-scale incident-response agent. It is designed to diagnose reproducible
service failures from logs, metrics, deployment history, and runbooks before proposing any
remediation.

The project starts with deterministic incident fixtures and typed, read-only diagnostic tools.
Later milestones will add a LangGraph workflow, durable checkpoints, human approval for risky
actions, hybrid retrieval, OpenTelemetry traces, and evaluation-driven fault injection.

## Why start without an LLM?

An agent cannot be evaluated if its environment changes every time it runs. The first milestone
therefore separates two concerns:

1. **Environment truth:** fixed incident evidence and known root causes.
2. **Agent behavior:** which tool it selects, which arguments it supplies, and which conclusion it
   reaches.

This makes future model and prompt experiments comparable instead of anecdotal.

## First milestone

- Three reproducible incident snapshots.
- Four read-only tools: `search_logs`, `query_metrics`, `get_deployments`, and `search_runbooks`.
- JSON-Schema-like contracts validated before handlers run.
- A registry that rejects unknown tools and malformed arguments.
- Standard-library unit tests with no API key or network access.

## Run the tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Planned architecture

```text
Alert -> Triage -> Parallel evidence collection -> Evidence fusion -> Diagnosis
                                                            |
                                      policy + human approval for risky actions
                                                            |
                                      execute -> verify -> compensate/escalate
```

No production system is modified in the current milestone. All evidence comes from local fixtures.


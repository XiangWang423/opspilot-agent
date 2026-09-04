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

## Second milestone: bounded diagnosis loop

- A `Policy` chooses either a structured `ToolCall` or a `FinalDiagnosis`.
- `AgentRunner` owns execution instead of allowing the policy to run code directly.
- Every successful or failed tool execution becomes an `Observation` for the next decision.
- Recoverable tool errors are returned to the policy so it can repair invalid arguments.
- A total step budget and repeated-call detection stop runaway loops and control cost.
- Deterministic policies test orchestration without network access or model tokens.

## Third milestone: OpenRouter policy adapter

- `OpenRouterPolicy` converts provider tool calls into the project's typed `ToolCall` decisions.
- Provider `call_id` values are preserved so every tool result is linked to the request that
  produced it.
- Model response validation rejects malformed tool arguments and invalid final diagnoses.
- Successfully queried evidence sources are removed from later tool choices to prevent redundant
  calls, and the final decision is reserved for producing a diagnosis.
- A command-line entry point assembles the repository, tools, policy, and bounded runner.
- Offline scripted-client tests cover the complete model -> tool -> observation -> model flow
  without spending API tokens.

## Run the tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Run one live diagnosis

Set the API key in the shell without putting it in source code or committing it:

```bash
export OPENROUTER_API_KEY="your-key"
```

Then diagnose a fixture incident with a small step budget:

```bash
PYTHONPATH=src python3 -m opspilot.cli inc-003 \
  --model openai/gpt-5-nano \
  --max-steps 4
```

Each policy decision may make one model API request. The step budget therefore bounds model
requests as well as runaway loops. Tool execution remains local and read-only.

## Planned architecture

```text
Alert -> Triage -> Parallel evidence collection -> Evidence fusion -> Diagnosis
                                                            |
                                      policy + human approval for risky actions
                                                            |
                                      execute -> verify -> compensate/escalate
```

No production system is modified in the current milestone. All evidence comes from local fixtures.

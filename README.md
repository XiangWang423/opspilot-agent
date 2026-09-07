# OpsPilot Agent

OpsPilot is a portfolio-scale incident-response agent. It is designed to diagnose reproducible
service failures from logs, metrics, deployment history, and runbooks before proposing any
remediation.

The project combines deterministic incident fixtures, typed read-only tools, a bounded agent loop,
an OpenRouter policy adapter, and a reproducible evaluation harness. Production-only extensions
such as durable checkpoints and human approval remain deliberately outside this portfolio MVP.

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
- The runner receives alert, service, severity, and start time, while fixture ground truth is kept
  outside model context to prevent evaluation leakage.
- Offline scripted-client tests cover the complete model -> tool -> observation -> model flow
  without spending API tokens.

## Fourth milestone: deterministic evaluation

- Every log and metric row has a stable evidence ID that can be compared with fixture truth.
- Each incident declares concrete diagnosis terms and the evidence records required to support it.
- The evaluator reports completion rate, diagnosis accuracy, evidence recall, grounded accuracy,
  average tool calls, and average steps.
- Correctness and grounding are separate: a plausible diagnosis is not counted as grounded unless
  the agent actually retrieved every required evidence record.
- One provider failure is recorded per case and does not abort the remaining evaluation suite.

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

## Evaluate all fixtures

Evaluation calls the configured model once per agent decision, so start with one incident:

```bash
PYTHONPATH=src python3 -m opspilot.evaluation \
  --incident inc-003 \
  --model openai/gpt-5-nano \
  --max-steps 6
```

Remove `--incident inc-003` to evaluate all three fixtures. The scorer is deterministic: it checks
declared root-cause terms and stable evidence IDs instead of paying a second LLM to judge prose.

## Architecture

```text
Alert -> Triage -> Parallel evidence collection -> Evidence fusion -> Diagnosis
                                                            |
                                      policy + human approval for risky actions
                                                            |
                                      execute -> verify -> compensate/escalate
```

No production system is modified. All evidence comes from local fixtures, every tool is read-only,
and every run is bounded. This is the completed portfolio MVP; checkpoints, approval-gated write
tools, hybrid retrieval, and OpenTelemetry are documented future production extensions rather than
half-implemented claims.

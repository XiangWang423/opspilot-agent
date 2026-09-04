from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .tools import ToolError, ToolRegistry


@dataclass(frozen=True)
class ToolCall:
    """A policy decision requesting one tool execution."""

    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Observation:
    """The recorded result of one tool execution."""

    tool_name: str
    arguments: dict[str, Any]
    output: Any | None = None
    error: str | None = None


@dataclass(frozen=True)
class FinalDiagnosis:
    """The policy's final incident diagnosis."""

    root_cause: str
    confidence: float


@dataclass
class AgentState:
    """Mutable investigation state passed back to the policy each step."""

    incident_id: str
    observations: list[Observation] = field(default_factory=list)
    steps: int = 0


@dataclass(frozen=True)
class AgentResult:
    status: Literal["completed", "failed"]
    diagnosis: FinalDiagnosis | None
    observations: tuple[Observation, ...]
    steps: int
    error: str | None = None


class Policy(Protocol):
    """Anything capable of choosing the next action from the current state."""

    def decide(
        self, state: AgentState, tool_specs: list[dict[str, Any]]
    ) -> ToolCall | FinalDiagnosis: ...


class AgentRunner:
    """Bounded loop that separates model decisions from tool execution."""

    def __init__(
        self,
        policy: Policy,
        tools: ToolRegistry,
        *,
        max_steps: int = 6,
        max_identical_tool_calls: int = 2,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if max_identical_tool_calls < 1:
            raise ValueError("max_identical_tool_calls must be at least 1")
        self.policy = policy
        self.tools = tools
        self.max_steps = max_steps
        self.max_identical_tool_calls = max_identical_tool_calls

    def run(self, incident_id: str) -> AgentResult:
        state = AgentState(incident_id=incident_id)
        previous_tool_call: ToolCall | None = None
        identical_call_count = 0

        for step in range(1, self.max_steps + 1):
            state.steps = step
            decision = self.policy.decide(state, self.tools.specifications)

            if isinstance(decision, FinalDiagnosis):
                return AgentResult(
                    status="completed",
                    diagnosis=decision,
                    observations=tuple(state.observations),
                    steps=state.steps,
                )

            if decision == previous_tool_call:
                identical_call_count += 1
            else:
                previous_tool_call = decision
                identical_call_count = 1

            if identical_call_count > self.max_identical_tool_calls:
                return AgentResult(
                    status="failed",
                    diagnosis=None,
                    observations=tuple(state.observations),
                    steps=state.steps,
                    error=(
                        "Repeated tool call detected: "
                        f"{decision.tool_name} was requested "
                        f"{identical_call_count} times consecutively"
                    ),
                )

            try:
                output = self.tools.invoke(decision.tool_name, decision.arguments)
                observation = Observation(
                    tool_name=decision.tool_name,
                    arguments=decision.arguments,
                    output=output,
                )
            except ToolError as exc:
                observation = Observation(
                    tool_name=decision.tool_name,
                    arguments=decision.arguments,
                    error=f"{type(exc).__name__}: {exc}",
                )

            state.observations.append(observation)

        return AgentResult(
            status="failed",
            diagnosis=None,
            observations=tuple(state.observations),
            steps=state.steps,
            error=f"Maximum step count ({self.max_steps}) reached",
        )

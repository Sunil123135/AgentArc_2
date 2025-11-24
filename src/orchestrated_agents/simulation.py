"""Deterministic simulation harness for the orchestrated agents stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import cycle
from typing import Callable, Iterable, Optional

import pandas as pd

from .coordinator import Coordinator
from .critic_agent import HeuristicCriticAgent
from .decision_agent import SimpleDecisionAgent
from .executor_adapter import StubExecutor
from .interfaces import Executor
from .memory_agent import SimpleMemoryAgent
from .models import (
    PlanStepKind,
    PlanStepStatus,
    SessionState,
    StrategyProfile,
    ToolPerfRecord,
)
from .perception_agent import RuleBasedPerceptionAgent
from .retriever_agent import DummyRetrieverAgent


HumanInputFn = Callable[[str, SessionState], str]


@dataclass
class SimulationResult:
    test_id: int
    query: str
    plan_summary: str
    result_summary: str
    step_failed_human_in_loop: bool
    tool_failed_human_in_loop: bool
    tool_names_used: list[str] = field(default_factory=list)
    final_status: str = "partial"


@dataclass
class ToolStats:
    tool_name: str
    success_count: int
    failure_count: int
    total_calls: int
    failure_rate: float


class SimulationExecutor(StubExecutor):
    """Stub executor that can deterministically fail steps/tools for simulation."""

    def __init__(
        self,
        responses: Optional[dict[str, str]] = None,
        fail_tool_name: Optional[str] = None,
        fail_first_execute: bool = False,
        fail_tool_on_first: bool = False,
    ):
        super().__init__(responses=responses)
        self.fail_tool_name = fail_tool_name
        self.fail_first_execute = fail_first_execute
        self.fail_tool_on_first = fail_tool_on_first
        self._failed_execute_step = False
        self._tool_failure_triggered = False

    def execute_step(self, step, session: SessionState):
        if step.kind == PlanStepKind.EXECUTE:
            if self.fail_tool_name and step.tool_name == self.fail_tool_name:
                err = SimulationToolFailureError(
                    f"Simulated tool '{step.tool_name}' outage"
                )
                raise err
            if self.fail_tool_on_first and not self._tool_failure_triggered:
                self._tool_failure_triggered = True
                raise SimulationToolFailureError(
                    "Simulated MCP tool channel unavailable"
                )
            if self.fail_first_execute and not self._failed_execute_step:
                self._failed_execute_step = True
                raise SimulationStepFailureError(
                    "Simulated complex step failure for demonstration"
                )
        return super().execute_step(step, session)


class SimulationToolFailureError(Exception):
    """Tagged exception for tool failures."""

    hil_category = "tool_failure"


class SimulationStepFailureError(Exception):
    """Tagged exception for complex step failures."""

    hil_category = "step_failure"


class SimulationCritic(HeuristicCriticAgent):
    """Critic that is lenient for successful steps to keep simulation flowing."""

    def review_result(
        self, step, perception, retrieval, session: SessionState
    ):
        report = super().review_result(step, perception, retrieval, session)
        if step.status == PlanStepStatus.SUCCESS:
            report.is_acceptable = True
            report.issues = []
            report.quality_score = 1.0
        return report


SIMULATION_QUERIES: list[dict[str, object]] = [
    {
        "query": "Summarize the revenue highlights for Q1.",
        "complex": False,
        "force_tool_fail": False,
        "force_step_fail": False,
    },
    {
        "query": "Compare two product launch campaigns and deliver insights.",
        "complex": False,
        "force_tool_fail": True,
        "force_step_fail": False,
    },
    {
        "query": (
            "Execute a python task that should trigger human escalation: "
            "download quarterly metrics, run Monte Carlo simulations, summarize anomalies."
        ),
        "complex": True,
        "force_tool_fail": False,
        "force_step_fail": True,
    },
    {
        "query": "Generate a concise plan for improving customer NPS via support tooling.",
        "complex": False,
        "force_tool_fail": True,
        "force_step_fail": False,
    },
    {
        "query": "Compile regulatory updates by region and outline required actions.",
        "complex": True,
        "force_tool_fail": False,
        "force_step_fail": True,
    },
]


def run_simulation(
    num_tests: int = 100,
    human_input_fn: Optional[HumanInputFn] = None,
) -> tuple[list[SimulationResult], list[ToolStats]]:
    """Run N synthetic tests and return row-wise + aggregated results."""
    if human_input_fn is None:
        human_input_fn = default_human_input_fn

    results: list[SimulationResult] = []
    all_tool_records: list[ToolPerfRecord] = []

    scenario_cycle = cycle(SIMULATION_QUERIES)

    for test_id in range(1, num_tests + 1):
        scenario = next(scenario_cycle)
        query = str(scenario["query"])
        is_complex = bool(scenario["complex"])
        force_tool_fail = bool(scenario["force_tool_fail"]) and test_id % 4 == 0
        force_step_fail = bool(scenario["force_step_fail"]) and test_id % 3 == 0

        executor = SimulationExecutor(
            fail_tool_name="python" if force_tool_fail else None,
            fail_first_execute=force_step_fail,
            fail_tool_on_first=force_tool_fail,
        )

        coordinator = Coordinator(
            perception_agent=RuleBasedPerceptionAgent(),
            retriever_agent=DummyRetrieverAgent(),
            memory_agent=SimpleMemoryAgent(),
            decision_agent=SimpleDecisionAgent(),
            critic_agent=SimulationCritic(),
            executor=executor,
            strategy=StrategyProfile.EXPLORATORY,
            human_input_callback=lambda prompt, session=None: human_input_fn(
                prompt, session  # type: ignore[arg-type]
            ),
        )

        metadata = {"is_complex_query": is_complex}
        session = coordinator.run(query, metadata=metadata)

        all_tool_records.extend(session.tool_performance.records)

        plan_summary = _summarize_plan(session)
        result_summary = (session.final_answer or "No final answer")[:160]
        step_hil = bool(session.meta.get("step_failed_human_in_loop"))
        tool_hil = bool(session.meta.get("tool_failed_human_in_loop"))
        tools_used = sorted(
            {
                record.tool_name
                for record in session.tool_performance.records
                if record.tool_name
            }
        )
        final_status = _determine_status(session)

        results.append(
            SimulationResult(
                test_id=test_id,
                query=query,
                plan_summary=plan_summary,
                result_summary=result_summary,
                step_failed_human_in_loop=step_hil,
                tool_failed_human_in_loop=tool_hil,
                tool_names_used=tools_used,
                final_status=final_status,
            )
        )

    tool_stats = _aggregate_tool_stats(all_tool_records)
    return results, tool_stats


def default_human_input_fn(prompt: str, session: Optional[SessionState] = None) -> str:
    """Return a canned response depending on prompt content."""
    prompt_lower = prompt.lower()
    if "tool" in prompt_lower:
        return "SIMULATED HUMAN: Acknowledge tool outage and use cached data."
    if "plan" in prompt_lower or "step" in prompt_lower:
        return "SIMULATED HUMAN: Break the task into smaller manual checkpoints."
    return "SIMULATED HUMAN: Proceed with caution."


def print_simulation_report(num_tests: int = 100) -> None:
    """Run the simulation and print formatted tables."""
    rows, tool_stats = run_simulation(num_tests=num_tests)
    df_main = pd.DataFrame(
        [
            {
                "TestID": r.test_id,
                "Query": r.query,
                "Plan": r.plan_summary,
                "Result": r.result_summary,
                "StepFail_HIL": r.step_failed_human_in_loop,
                "MCPFail_HIL": r.tool_failed_human_in_loop,
                "ToolsUsed": ", ".join(r.tool_names_used),
                "FinalStatus": r.final_status,
            }
            for r in rows
        ]
    )

    df_tools = pd.DataFrame(
        [
            {
                "Tool": stat.tool_name,
                "Success": stat.success_count,
                "Failure": stat.failure_count,
                "Total": stat.total_calls,
                "FailureRate": round(stat.failure_rate, 3),
            }
            for stat in tool_stats
        ]
    ).sort_values(by="Failure", ascending=False)

    print("=== Simulation Results ({} Tests) ===".format(len(df_main)))
    print(df_main.to_string(index=False))
    print("\n=== Tool Usage Statistics ===")
    print(df_tools.to_string(index=False))


def _summarize_plan(session: SessionState) -> str:
    """Create a short textual summary of the plan steps."""
    plan = session.get_active_plan() or (session.plans[-1] if session.plans else None)
    if not plan:
        return "no-plan"
    return " -> ".join(step.kind.value for step in plan.steps)


def _determine_status(session: SessionState) -> str:
    if session.done and session.final_answer:
        return "success"
    if session.done:
        return "partial"
    return "failed"


def _aggregate_tool_stats(records: Iterable[ToolPerfRecord]) -> list[ToolStats]:
    stats: dict[str, ToolStats] = {}
    for record in records:
        tool = record.tool_name or "unknown"
        if tool not in stats:
            stats[tool] = ToolStats(
                tool_name=tool,
                success_count=0,
                failure_count=0,
                total_calls=0,
                failure_rate=0.0,
            )
        entry = stats[tool]
        entry.total_calls += 1
        if record.success:
            entry.success_count += 1
        else:
            entry.failure_count += 1
    for entry in stats.values():
        total = max(entry.total_calls, 1)
        entry.failure_rate = entry.failure_count / total
    return list(stats.values())


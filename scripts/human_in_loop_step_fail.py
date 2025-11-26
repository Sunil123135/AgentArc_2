"""Human-In-Loop demo for step-level failures and plan failures."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# --- Path setup so we can import orchestrated_agents from src/ ----------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from orchestrated_agents.coordinator import Coordinator
from orchestrated_agents.critic_agent import HeuristicCriticAgent
from orchestrated_agents.decision_agent import SimpleDecisionAgent
from orchestrated_agents.memory_agent import SimpleMemoryAgent
from orchestrated_agents.models import (
    StrategyProfile,
    SessionState,
    PlanStep,
    PlanStepKind,
    ToolPerfRecord,
)
from orchestrated_agents.perception_agent import RuleBasedPerceptionAgent
from orchestrated_agents.retriever_agent import DummyRetrieverAgent
from orchestrated_agents.interfaces import Executor


# ---------------------------------------------------------------------
# Custom exception to signal *step* failure to Coordinator
# ---------------------------------------------------------------------
class StepFailureError(Exception):
    """Exception used to trigger Coordinator's step_failure human-in-loop path."""

    def __init__(self, message: str, hil_category: str = "step_failure") -> None:
        super().__init__(message)
        # Coordinator.run checks this attribute via getattr(e, "hil_category", None)
        self.hil_category = hil_category


# ---------------------------------------------------------------------
# Local executor that simulates step failures (not tool outages)
# ---------------------------------------------------------------------
class StepFailingExecutor(Executor):
    """
    Executor that *always* reports a step-level failure for any step
    that reaches it. This guarantees a Human-In-Loop step escalation
    for the demo.
    """

    def __init__(self) -> None:
        # Track how many times each step.id has been executed (for logging)
        self.attempts: Dict[str, int] = {}

    def execute_step(
        self,
        step: PlanStep,
        session: SessionState,
    ) -> Tuple[str, Dict[str, Any], ToolPerfRecord]:
        step_key = step.id
        self.attempts[step_key] = self.attempts.get(step_key, 0) + 1
        attempt_no = self.attempts[step_key]

        # We *always* raise a step-level failure here.
        msg = (
            f"Simulated step-level failure for step {step.id} "
            f"(kind={step.kind.value}, attempt={attempt_no})."
        )
        raise StepFailureError(msg, hil_category="step_failure")


# ---------------------------------------------------------------------
# Human-In-Loop callback used by Coordinator
# ---------------------------------------------------------------------
def human_input_callback(prompt: str, session: SessionState) -> str:
    """
    Human-In-Loop callback used by Coordinator._handle_tool_failure,
    _handle_step_failure, and _trigger_human_in_loop.
    """
    print("\n=== HUMAN-IN-LOOP TRIGGERED ===")
    print(f"Prompt from coordinator: {prompt}")

    escalation_type = "plan_failure"
    failed_tool: Optional[str] = None

    if prompt.startswith("Tool "):
        escalation_type = "tool_failure"
        parts = prompt.split(" failed", 1)
        if parts:
            failed_tool = parts[0].replace("Tool ", "").strip()
    elif prompt.startswith("Step '"):
        escalation_type = "step_failure"
    else:
        escalation_type = "plan_failure"

    print(f"Escalation type inferred from prompt: {escalation_type}")
    if failed_tool:
        print(f"Detected failed tool in HIL callback: {failed_tool}")

    # Track failed tools seen in human-in-loop (for extra safety)
    if failed_tool:
        failed_list: List[str] = session.meta.get("failed_tools_seen_in_hil", [])
        if failed_tool not in failed_list:
            failed_list.append(failed_tool)
        session.meta["failed_tools_seen_in_hil"] = failed_list

    if escalation_type == "tool_failure" and failed_tool:
        simulated_answer = (
            f"SIMULATED HUMAN: Tool '{failed_tool}' failed. "
            f"Use default metrics and proceed to summarize anomalies."
        )
    elif escalation_type == "step_failure":
        simulated_answer = (
            "SIMULATED HUMAN: For this failed step, simplify the task into "
            "smaller sub-steps and retry with fewer assumptions."
        )
    else:
        simulated_answer = (
            "SIMULATED HUMAN: Acknowledge plan escalation. Focus on the key goal, "
            "reduce scope, and retry with a simpler plan."
        )

    print(f"Simulated human answer: {simulated_answer}\n")
    return simulated_answer


# ---------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------
def run_demo() -> None:
    print(">>> Starting Human-In-Loop step-failure demo...\n")

    perception_agent = RuleBasedPerceptionAgent()
    retriever_agent = DummyRetrieverAgent()
    memory_agent = SimpleMemoryAgent()
    decision_agent = SimpleDecisionAgent()
    critic_agent = HeuristicCriticAgent()

    failing_executor = StepFailingExecutor()

    coordinator = Coordinator(
        perception_agent=perception_agent,
        retriever_agent=retriever_agent,
        memory_agent=memory_agent,
        decision_agent=decision_agent,
        critic_agent=critic_agent,
        executor=failing_executor,
        strategy=StrategyProfile.CONSERVATIVE,
        human_input_callback=human_input_callback,
    )

    # Mark this as a complex query (optional but aligns with assignment semantics)
    metadata = {"is_complex_query": True}

    # Any reasonably complex query is fine; executor will fail the step anyway.
    query = (
        "For a complex analytics workflow, execute multiple reasoning steps: "
        "gather quarterly metrics, run statistical tests, and summarize risks. "
        "This should demonstrate Human-In-Loop for a failed step."
    )

    try:
        session = coordinator.run(query, metadata=metadata)
    except Exception as e:
        # If something unexpected blows up, we still get output.
        print("\n*** Unexpected exception from coordinator.run() ***")
        print(repr(e))
        return

    print("\n=== Final Answer ===")
    print(session.final_answer)

    print("\n=== Human-In-Loop Events ===")
    hil_events: List[Dict[str, Any]] = session.meta.get("human_in_loop_events", [])
    if not hil_events:
        print("No human-in-loop events recorded.")
    else:
        for idx, ev in enumerate(hil_events, start=1):
            print(
                f"[{idx}] category={ev.get('category')} "
                f"step_id={ev.get('step_id')} "
                f"turn={ev.get('turn')}"
            )
            print(f"    prompt:   {ev.get('prompt')}")
            print(f"    response: {ev.get('response')}")
            print()

    print("\n=== Step Failures Triggering Human-In-Loop ===")
    failed_steps: Set[str] = set()

    for event in hil_events:
        if event.get("category") == "step_failure" and event.get("step_id"):
            failed_steps.add(event["step_id"])

    if failed_steps:
        print("Steps that failed and triggered human-in-loop (by ID):")
        for sid in sorted(failed_steps):
            print(f"  - {sid}")
    else:
        print(
            "No step_failure events detected "
            "(only tool_failure or plan_failure may have occurred)."
        )

    
if __name__ == "__main__":
    run_demo()

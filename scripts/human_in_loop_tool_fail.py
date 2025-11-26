"""Human-In-Loop demo showing tool_failure and plan_failure paths."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------
# Make src/ importable so we can import orchestrated_agents.*
# ---------------------------------------------------------------------
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
# Custom exception to signal tool failure to Coordinator
# ---------------------------------------------------------------------
class ToolFailureError(Exception):
    """Exception used to trigger Coordinator's tool_failure human-in-loop path."""

    def __init__(self, message: str, hil_category: str = "tool_failure") -> None:
        super().__init__(message)
        # Coordinator.run checks this attribute via getattr(e, "hil_category", None)
        self.hil_category = hil_category


# ---------------------------------------------------------------------
# Local executor that simulates MCP-style tool failures
# ---------------------------------------------------------------------
class HilFailingExecutor(Executor):
    """
    Executor that fails specific tools on first attempt and sets hil_category='tool_failure'
    so Coordinator._handle_tool_failure(...) is invoked.
    """

    def __init__(self, failing_tools: Optional[Set[str]] = None) -> None:
        self.failing_tools: Set[str] = failing_tools or set()
        self.attempts: Dict[str, int] = {}

    def execute_step(
        self, step: PlanStep, session: SessionState
    ) -> Tuple[str, Dict[str, Any], ToolPerfRecord]:
        # Non-EXECUTE steps: behave like other executors and just succeed with dummy output
        if step.kind != PlanStepKind.EXECUTE:
            return (
                "Step completed",
                {},
                ToolPerfRecord(
                    tool_name="none",
                    success=True,
                    latency_ms=10,
                    step_id=step.id,
                ),
            )

        tool_name = step.tool_name or "unknown"
        attempt_key = f"{step.id}_{tool_name}"
        self.attempts[attempt_key] = self.attempts.get(attempt_key, 0) + 1

        # Force failure on first attempt for selected tools
        if tool_name in self.failing_tools and self.attempts[attempt_key] == 1:
            # This message will be passed as error_message into _handle_tool_failure
            error_message = f"Simulated MCP outage: {tool_name} unreachable."
            raise ToolFailureError(error_message, hil_category="tool_failure")

        # On retries or non-failing tools, simulate success
        result_text = (
            f"Executed {tool_name} successfully (attempt {self.attempts[attempt_key]})"
        )
        result_payload: Dict[str, Any] = {
            "tool": tool_name,
            "args": step.tool_args,
            "attempt": self.attempts[attempt_key],
        }

        record = ToolPerfRecord(
            tool_name=tool_name,
            success=True,
            latency_ms=50,
            step_id=step.id,
        )

        return result_text, result_payload, record


# ---------------------------------------------------------------------
# Human-in-loop callback used by Coordinator
# ---------------------------------------------------------------------
def human_input_callback(prompt: str, session: SessionState) -> str:
    """
    Human-In-Loop callback used by:
      - _handle_tool_failure  (category='tool_failure')
      - _handle_step_failure  (category='step_failure')
      - _trigger_human_in_loop (category='plan_failure')

    We infer escalation type from the prompt shape:
      - Tool failure: "Tool <name> failed with error '...'. Provide manual guidance..."
      - Step failure: "Step '<description>' failed (...). Provide manual advice..."
      - Plan failure: "Plan escalation triggered for reason: ... Suggest next action..."
    """
    print("\n=== HUMAN-IN-LOOP TRIGGERED ===")
    print(f"Prompt from coordinator: {prompt}")

    escalation_type = "plan_failure"
    failed_tool: Optional[str] = None

    if prompt.startswith("Tool "):
        escalation_type = "tool_failure"
        # Parse "Tool <name> failed with error '...'"
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

    # Track failed tools seen in HIL for reporting
    if failed_tool:
        failed_list: List[str] = session.meta.get("failed_tools_seen_in_hil", [])
        if failed_tool not in failed_list:
            failed_list.append(failed_tool)
        session.meta["failed_tools_seen_in_hil"] = failed_list

    # Build simulated human answer based on escalation type
    if escalation_type == "tool_failure" and failed_tool:
        simulated_answer = (
            f"SIMULATED HUMAN: Tool '{failed_tool}' failed. "
            f"Use default metrics and proceed to summarize anomalies."
        )
    elif escalation_type == "step_failure":
        simulated_answer = (
            "SIMULATED HUMAN: For this failed step, break the task into smaller "
            "sub-goals and retry with a simpler approach."
        )
    else:  # plan_failure or unknown
        simulated_answer = (
            "SIMULATED HUMAN: Acknowledge plan escalation. Focus on the main goal, "
            "reduce complexity, and retry with a shorter plan."
        )

    print(f"Simulated human answer: {simulated_answer}\n")
    return simulated_answer


# ---------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------
def run_demo() -> None:
    """Run a session where a tool failure triggers a Human-In-Loop step."""
    perception_agent = RuleBasedPerceptionAgent()
    retriever_agent = DummyRetrieverAgent()
    memory_agent = SimpleMemoryAgent()
    decision_agent = SimpleDecisionAgent()
    critic_agent = HeuristicCriticAgent()

    # Choose tool names that your DecisionAgent actually uses.
    # Often this will be "python" and/or "calculator".
    failing_tools: Set[str] = {"python"}  # you can also add "calculator" here
    executor = HilFailingExecutor(failing_tools=failing_tools)

    coordinator = Coordinator(
        perception_agent=perception_agent,
        retriever_agent=retriever_agent,
        memory_agent=memory_agent,
        decision_agent=decision_agent,
        critic_agent=critic_agent,
        executor=executor,
        strategy=StrategyProfile.CONSERVATIVE,
        human_input_callback=human_input_callback,
    )

    query = (
        "Execute a python task that should trigger human escalation: "
        "download quarterly metrics, run a Monte Carlo simulation, "
        "and summarize anomalies."
    )

    session = coordinator.run(query)

    print("\n=== Final Answer ===")
    print(session.final_answer or "No final answer generated.")

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

    print("\n=== Failed Tool Names (derived from tool_failure HIL events) ===")
    failed_tools_detected: Set[str] = set()

    # Derive failed tools from tool_failure events
    for event in hil_events:
        if event.get("category") == "tool_failure":
            prompt = event.get("prompt") or ""
            tool_name: Optional[str] = None
            if prompt.startswith("Tool "):
                parts = prompt.split(" failed", 1)
                if parts:
                    tool_name = parts[0].replace("Tool ", "").strip()
            if tool_name:
                failed_tools_detected.add(tool_name)

    # Also include ones we stored in meta["failed_tools_seen_in_hil"]
    for t in session.meta.get("failed_tools_seen_in_hil", []):
        failed_tools_detected.add(t)

    if failed_tools_detected:
        print(
            "Tools that failed and triggered human-in-loop: "
            f"{', '.join(sorted(failed_tools_detected))}"
        )
    else:
        print(
            "No tool failures detected in human-in-loop events "
            "(only plan/step failures may have occurred)."
        )


if __name__ == "__main__":
    run_demo()

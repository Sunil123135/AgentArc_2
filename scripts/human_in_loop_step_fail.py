"""Human-In-Loop demo for complex plan step failure."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from orchestrated_agents.coordinator import Coordinator
from orchestrated_agents.critic_agent import HeuristicCriticAgent
from orchestrated_agents.decision_agent import SimpleDecisionAgent
from orchestrated_agents.executor_adapter import StubExecutor
from orchestrated_agents.memory_agent import SimpleMemoryAgent
from orchestrated_agents.models import StrategyProfile
from orchestrated_agents.perception_agent import RuleBasedPerceptionAgent
from orchestrated_agents.retriever_agent import DummyRetrieverAgent
from orchestrated_agents.models import PlanStepKind


class ScenarioTweakedExecutor(StubExecutor):
    """Custom executor to fail a specific complex step to simulate escalation."""

    def execute_step(self, step, session):
        if step.kind == PlanStepKind.EXECUTE and "Monte Carlo" in step.description:
            raise Exception("Monte Carlo library missing dependencies")
        return super().execute_step(step, session)


def run_demo() -> None:
    """Run a session where a complex step fail triggers Human-In-Loop."""
    perception_agent = RuleBasedPerceptionAgent()
    retriever_agent = DummyRetrieverAgent()
    memory_agent = SimpleMemoryAgent()
    decision_agent = SimpleDecisionAgent()
    critic_agent = HeuristicCriticAgent()
    executor = ScenarioTweakedExecutor()

    coordinator = Coordinator(
        perception_agent=perception_agent,
        retriever_agent=retriever_agent,
        memory_agent=memory_agent,
        decision_agent=decision_agent,
        critic_agent=critic_agent,
        executor=executor,
        strategy=StrategyProfile.CONSERVATIVE,
    )

    query = (
        "Prepare a monthly performance review: "
        "aggregate international revenue, run Monte Carlo risk modeling, "
        "and summarize issues for the board."
    )

    session = coordinator.run(query)

    print("\n=== Final Answer ===")
    print(session.final_answer)

    print("\n=== Human-In-Loop Metadata ===")
    print(session.meta.get("human_in_loop_escalations"))


if __name__ == "__main__":
    run_demo()


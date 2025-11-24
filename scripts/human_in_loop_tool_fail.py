"""Human-In-Loop demo for persistent tool failures."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from orchestrated_agents.coordinator import Coordinator
from orchestrated_agents.critic_agent import HeuristicCriticAgent
from orchestrated_agents.decision_agent import SimpleDecisionAgent
from orchestrated_agents.executor_adapter import FailingExecutor
from orchestrated_agents.memory_agent import SimpleMemoryAgent
from orchestrated_agents.models import StrategyProfile
from orchestrated_agents.perception_agent import RuleBasedPerceptionAgent
from orchestrated_agents.retriever_agent import DummyRetrieverAgent


def run_demo() -> None:
    """Run a session where tool failure forces Human-In-Loop escalation."""
    perception_agent = RuleBasedPerceptionAgent()
    retriever_agent = DummyRetrieverAgent()
    memory_agent = SimpleMemoryAgent()
    decision_agent = SimpleDecisionAgent()
    critic_agent = HeuristicCriticAgent()

    failing_executor = FailingExecutor(failing_tools={"python"})

    coordinator = Coordinator(
        perception_agent=perception_agent,
        retriever_agent=retriever_agent,
        memory_agent=memory_agent,
        decision_agent=decision_agent,
        critic_agent=critic_agent,
        executor=failing_executor,
        strategy=StrategyProfile.CONSERVATIVE,
    )

    query = (
        "Execute a python task that triggers human escalation: "
        "download quarterly metrics, run a Monte Carlo simulation, summarize anomalies."
    )

    session = coordinator.run(query)

    print("\n=== Final Answer ===")
    print(session.final_answer)

    print("\n=== Human-In-Loop Metadata ===")
    print(session.meta.get("human_in_loop_escalations"))


if __name__ == "__main__":
    run_demo()


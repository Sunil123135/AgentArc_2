"""Tests for tool performance log persistence and ingestion."""

from pathlib import Path

from orchestrated_agents.coordinator import Coordinator
from orchestrated_agents.critic_agent import HeuristicCriticAgent
from orchestrated_agents.decision_agent import SimpleDecisionAgent
from orchestrated_agents.executor_adapter import StubExecutor
from orchestrated_agents.memory_agent import SimpleMemoryAgent
from orchestrated_agents.models import StrategyProfile, ToolPerformanceLog
from orchestrated_agents.perception_agent import RuleBasedPerceptionAgent
from orchestrated_agents.retriever_agent import DummyRetrieverAgent


def test_tool_performance_log_persist_and_feed(tmp_path):
    """Coordinator should save tool logs and feed summary into memory agent."""
    perception_agent = RuleBasedPerceptionAgent()
    retriever_agent = DummyRetrieverAgent()
    memory_agent = SimpleMemoryAgent()
    decision_agent = SimpleDecisionAgent()
    critic_agent = HeuristicCriticAgent()
    executor = StubExecutor(
        responses={
            "python": "Python step done",
            "calculator": "Calculation done",
            "web_search": "Search results delivered",
        }
    )

    log_dir = tmp_path / "tool_logs"
    coordinator = Coordinator(
        perception_agent=perception_agent,
        retriever_agent=retriever_agent,
        memory_agent=memory_agent,
        decision_agent=decision_agent,
        critic_agent=critic_agent,
        executor=executor,
        strategy=StrategyProfile.EXPLORATORY,
        tool_log_dir=log_dir,
    )

    session = coordinator.run(
        "Calculate 1+1 and calculate 2+2 and write a python script"
    )

    assert session.done
    summary = session.tool_performance.summarize_by_tool()
    assert summary["calculator"]["total_calls"] == 2
    assert summary["python"]["total_calls"] == 1

    log_path = Path(session.meta["tool_performance_log_path"])
    assert log_path.exists()
    persisted = ToolPerformanceLog.load_from_path(log_path)
    assert persisted.model_dump() == session.tool_performance.model_dump()

    notes_lower = (session.memory_state.notes or "").lower()
    assert "tool performance summary" in notes_lower
    assert "calculator" in notes_lower
    assert "python" in notes_lower


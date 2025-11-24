"""Tests for coordinator happy path."""

import sys
from pathlib import Path

# Add src directory to Python path (for direct execution)
# When using pytest, conftest.py handles this automatically
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from orchestrated_agents.coordinator import Coordinator
from orchestrated_agents.critic_agent import HeuristicCriticAgent
from orchestrated_agents.decision_agent import SimpleDecisionAgent
from orchestrated_agents.executor_adapter import StubExecutor
from orchestrated_agents.memory_agent import SimpleMemoryAgent
from orchestrated_agents.models import StrategyProfile
from orchestrated_agents.perception_agent import RuleBasedPerceptionAgent
from orchestrated_agents.retriever_agent import DummyRetrieverAgent


def test_coordinator_happy_path():
    """Test coordinator produces final answer and steps move from pending to success."""
    # Create all agents
    perception_agent = RuleBasedPerceptionAgent()
    retriever_agent = DummyRetrieverAgent()
    memory_agent = SimpleMemoryAgent()
    decision_agent = SimpleDecisionAgent()
    critic_agent = HeuristicCriticAgent()
    executor = StubExecutor(
        responses={
            "python": "Task completed successfully",
        }
    )
    
    coordinator = Coordinator(
        perception_agent=perception_agent,
        retriever_agent=retriever_agent,
        memory_agent=memory_agent,
        decision_agent=decision_agent,
        critic_agent=critic_agent,
        executor=executor,
        strategy=StrategyProfile.CONSERVATIVE,
    )
    
    # Run coordinator
    session = coordinator.run("Calculate 2 + 2")
    
    # Assertions
    assert session.done
    assert session.final_answer is not None
    assert len(session.plans) > 0
    
    # Check that plan has steps
    plan = session.get_active_plan()
    if plan:
        assert len(plan.steps) > 0
        
        # Check that at least some steps moved to success
        success_steps = [
            s for s in plan.steps if s.status.value == "success"
        ]
        assert len(success_steps) > 0, "At least one step should be successful"
    
    # Check perception snapshots were created
    assert len(session.perception_snapshots) > 0


def test_coordinator_with_multiple_steps():
    """Test coordinator with a query that requires multiple steps."""
    perception_agent = RuleBasedPerceptionAgent()
    retriever_agent = DummyRetrieverAgent()
    memory_agent = SimpleMemoryAgent()
    decision_agent = SimpleDecisionAgent()
    critic_agent = HeuristicCriticAgent()
    executor = StubExecutor()
    
    coordinator = Coordinator(
        perception_agent=perception_agent,
        retriever_agent=retriever_agent,
        memory_agent=memory_agent,
        decision_agent=decision_agent,
        critic_agent=critic_agent,
        executor=executor,
        strategy=StrategyProfile.EXPLORATORY,  # More steps allowed
    )
    
    session = coordinator.run("Analyze data and then summarize results")
    
    assert session.done
    assert session.final_answer is not None
    
    # Check that multiple steps were executed
    plan = session.get_active_plan()
    if plan:
        executed_steps = [
            s for s in plan.steps
            if s.status.value in ["success", "failed"]
        ]
        assert len(executed_steps) > 0


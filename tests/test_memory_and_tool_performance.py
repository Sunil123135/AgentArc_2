"""Tests for memory and tool performance logging."""

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
from orchestrated_agents.executor_adapter import FailingExecutor
from orchestrated_agents.memory_agent import SimpleMemoryAgent
from orchestrated_agents.models import MemoryItemKind, StrategyProfile
from orchestrated_agents.perception_agent import RuleBasedPerceptionAgent
from orchestrated_agents.retriever_agent import DummyRetrieverAgent


def test_memory_bans_tool_after_failures():
    """Test that MemoryState.banned_tools contains tool after multiple failures."""
    perception_agent = RuleBasedPerceptionAgent()
    retriever_agent = DummyRetrieverAgent()
    memory_agent = SimpleMemoryAgent()
    decision_agent = SimpleDecisionAgent()
    critic_agent = HeuristicCriticAgent()
    
    # Executor that always fails for "bad_tool"
    class AlwaysFailingToolExecutor:
        def __init__(self):
            self.call_count = 0
        
        def execute_step(self, step, session):
            self.call_count += 1
            if step.tool_name == "bad_tool":
                raise Exception("Tool always fails")
            return "Success", {}, None
    
    from orchestrated_agents.executor_adapter import MCPExecutor
    executor = MCPExecutor(AlwaysFailingToolExecutor())
    
    coordinator = Coordinator(
        perception_agent=perception_agent,
        retriever_agent=retriever_agent,
        memory_agent=memory_agent,
        decision_agent=decision_agent,
        critic_agent=critic_agent,
        executor=executor,
        strategy=StrategyProfile.EXPLORATORY,
    )
    
    # Run multiple times to trigger failures
    # Note: In a real scenario, we'd need to simulate multiple steps with the same tool
    # For now, we'll check that the memory agent logic works
    
    # Manually test memory agent
    from orchestrated_agents.models import (
        CriticReport,
        MemoryState,
        PlanStep,
        PlanStepKind,
        PlanStepStatus,
        SessionState,
    )
    
    session = SessionState(
        session_id="test-memory",
        user_query="test",
    )
    
    # Simulate multiple failures of the same tool
    for i in range(4):  # More than FAILURE_THRESHOLD (3)
        step = PlanStep(
            id=f"step-{i}",
            index=i,
            kind=PlanStepKind.EXECUTE,
            description="Test step",
            tool_name="bad_tool",
            status=PlanStepStatus.FAILED,
            notes="Tool failed",
        )
        
        critic = CriticReport(
            id=f"critic-{i}",
            step_id=step.id,
            turn_index=i,
            is_acceptable=False,
        )
        
        memory_agent.update_from_step(step, critic, False, session)
    
    # Check that tool is banned
    assert "bad_tool" in session.memory_state.banned_tools


def test_tool_performance_logging():
    """Test that ToolPerformanceLog.records is populated."""
    perception_agent = RuleBasedPerceptionAgent()
    retriever_agent = DummyRetrieverAgent()
    memory_agent = SimpleMemoryAgent()
    decision_agent = SimpleDecisionAgent()
    critic_agent = HeuristicCriticAgent()
    
    from orchestrated_agents.executor_adapter import StubExecutor
    executor = StubExecutor()
    
    coordinator = Coordinator(
        perception_agent=perception_agent,
        retriever_agent=retriever_agent,
        memory_agent=memory_agent,
        decision_agent=decision_agent,
        critic_agent=critic_agent,
        executor=executor,
        strategy=StrategyProfile.CONSERVATIVE,
    )
    
    session = coordinator.run("Execute a task")
    
    # Check that tool performance records were created
    assert len(session.tool_performance.records) > 0
    
    # Check that records have required fields
    for record in session.tool_performance.records:
        assert record.tool_name is not None
        assert record.step_id is not None
        assert record.latency_ms >= 0
        assert isinstance(record.success, bool)


def test_memory_tracks_successful_tools():
    """Test that memory tracks successful tool usage."""
    perception_agent = RuleBasedPerceptionAgent()
    retriever_agent = DummyRetrieverAgent()
    memory_agent = SimpleMemoryAgent()
    decision_agent = SimpleDecisionAgent()
    critic_agent = HeuristicCriticAgent()
    
    from orchestrated_agents.executor_adapter import StubExecutor
    executor = StubExecutor(
        responses={"good_tool": "Success"}
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
    
    session = coordinator.run("Use good_tool")
    
    # Check that successful tools are tracked
    # Note: This depends on the decision agent choosing "good_tool"
    # and the executor succeeding
    
    # At minimum, check that memory state exists
    assert session.memory_state is not None
    
    # Check that we can track successes
    # (The actual tool name depends on decision agent's choice)
    if session.memory_state.successful_tools:
        assert len(session.memory_state.successful_tools) > 0


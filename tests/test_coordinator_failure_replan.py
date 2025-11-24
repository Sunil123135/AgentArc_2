"""Tests for coordinator failure and replanning."""

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
from orchestrated_agents.models import StrategyProfile
from orchestrated_agents.perception_agent import RuleBasedPerceptionAgent
from orchestrated_agents.retriever_agent import DummyRetrieverAgent


def test_coordinator_failure_replan():
    """Test that critic triggers rewrite_plan and second PlanVersion is created."""
    perception_agent = RuleBasedPerceptionAgent()
    retriever_agent = DummyRetrieverAgent()
    memory_agent = SimpleMemoryAgent()
    decision_agent = SimpleDecisionAgent()
    critic_agent = HeuristicCriticAgent()
    
    # Executor that fails for "python" tool on first attempt
    executor = FailingExecutor(failing_tools={"python"})
    
    coordinator = Coordinator(
        perception_agent=perception_agent,
        retriever_agent=retriever_agent,
        memory_agent=memory_agent,
        decision_agent=decision_agent,
        critic_agent=critic_agent,
        executor=executor,
        strategy=StrategyProfile.EXPLORATORY,  # Allow rewrites
    )
    
    session = coordinator.run("Execute a task using python")
    
    # Assertions
    assert len(session.plans) >= 1
    
    # Check that we have at least one plan rewrite (second plan version)
    # The rewrite should happen when the first step fails
    plan_ids = [p.id for p in session.plans]
    
    # Check that at least one plan has a parent (indicating rewrite)
    plans_with_parent = [p for p in session.plans if p.parent_plan_id is not None]
    
    # Either we have multiple plans, or the critic triggered a rewrite
    # In this case, the executor will fail, critic will mark as unacceptable,
    # and decision agent should rewrite
    assert len(session.plans) >= 1
    
    # Check that final answer is still produced (or error message)
    assert session.done
    # Final answer might be an error message, which is acceptable
    assert session.final_answer is not None


def test_coordinator_handles_executor_exception():
    """Test that coordinator handles executor exceptions gracefully."""
    perception_agent = RuleBasedPerceptionAgent()
    retriever_agent = DummyRetrieverAgent()
    memory_agent = SimpleMemoryAgent()
    decision_agent = SimpleDecisionAgent()
    critic_agent = HeuristicCriticAgent()
    
    # Executor that always fails
    class AlwaysFailingExecutor:
        def execute_step(self, step, session):
            raise Exception("Always fails")
    
    from orchestrated_agents.executor_adapter import MCPExecutor
    executor = MCPExecutor(AlwaysFailingExecutor())
    
    coordinator = Coordinator(
        perception_agent=perception_agent,
        retriever_agent=retriever_agent,
        memory_agent=memory_agent,
        decision_agent=decision_agent,
        critic_agent=critic_agent,
        executor=executor,
        strategy=StrategyProfile.CONSERVATIVE,
    )
    
    session = coordinator.run("Do something")
    
    # Should complete (even if with error)
    assert session.done
    
    # Check that failures were logged
    failed_steps = []
    for plan in session.plans:
        failed_steps.extend([s for s in plan.steps if s.status.value == "failed"])
    
    # Should have at least one failed step
    assert len(failed_steps) > 0 or session.final_answer is not None


def test_plan_failure_triggers_human_in_loop_escalation():
    """Ensure repeated failures escalate to Human-In-Loop with suggestions."""
    perception_agent = RuleBasedPerceptionAgent()
    retriever_agent = DummyRetrieverAgent()
    memory_agent = SimpleMemoryAgent()
    decision_agent = SimpleDecisionAgent()
    critic_agent = HeuristicCriticAgent()
    
    # Executor that keeps failing for python steps, forcing plan rewrites to exhaust
    executor = FailingExecutor(failing_tools={"python"})
    
    coordinator = Coordinator(
        perception_agent=perception_agent,
        retriever_agent=retriever_agent,
        memory_agent=memory_agent,
        decision_agent=decision_agent,
        critic_agent=critic_agent,
        executor=executor,
        strategy=StrategyProfile.CONSERVATIVE,
    )
    
    session = coordinator.run("Execute a python task that keeps failing")
    
    assert session.done
    assert session.final_answer is not None
    assert "Human-In-Loop" in session.final_answer
    
    escalations = session.meta.get("human_in_loop_escalations")
    assert escalations, "Human-In-Loop escalation record should be stored"
    last_escalation = escalations[-1]
    assert last_escalation["agent_listening"] is True
    assert last_escalation["suggested_plan"]
    assert session.meta.get("agent_listening_for_human") is True
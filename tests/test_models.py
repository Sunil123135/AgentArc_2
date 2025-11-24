"""Tests for models."""

import sys
from pathlib import Path

# Add src directory to Python path (for direct execution)
# When using pytest, conftest.py handles this automatically
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from orchestrated_agents.models import (
    MemoryState,
    PlanStep,
    PlanStepKind,
    PlanStepStatus,
    PlanVersion,
    PlanStatus,
    SessionState,
    StrategyProfile,
    ToolPerfRecord,
)


def test_session_state_mark_done():
    """Test that SessionState.mark_done sets done and final_answer."""
    session = SessionState(
        session_id="test-1",
        user_query="test query",
    )
    
    assert not session.done
    assert session.final_answer is None
    
    session.mark_done("Test answer")
    
    assert session.done
    assert session.final_answer == "Test answer"
    
    # Check that active plan is marked as completed
    plan = PlanVersion(
        id="plan-1",
        created_turn=0,
        steps=[],
        status=PlanStatus.ACTIVE,
    )
    session.set_active_plan(plan)
    session.mark_done("Another answer")
    
    assert session.get_active_plan().status == PlanStatus.COMPLETED


def test_plan_version_helpers():
    """Test PlanVersion helper methods."""
    steps = [
        PlanStep(
            id=f"step-{i}",
            index=i,
            kind=PlanStepKind.EXECUTE,
            description=f"Step {i}",
            status=PlanStepStatus.PENDING,
        )
        for i in range(3)
    ]
    
    plan = PlanVersion(
        id="plan-1",
        created_turn=0,
        steps=steps,
        current_index=0,
    )
    
    # Test current_step
    current = plan.current_step()
    assert current is not None
    assert current.index == 0
    
    # Test advance
    plan.advance()
    assert plan.current_index == 1
    current = plan.current_step()
    assert current.index == 1
    
    # Test replace_remaining_steps
    new_steps = [
        PlanStep(
            id="step-new-1",
            index=0,  # Will be re-indexed
            kind=PlanStepKind.EXECUTE,
            description="New step 1",
            status=PlanStepStatus.PENDING,
        ),
        PlanStep(
            id="step-new-2",
            index=0,  # Will be re-indexed
            kind=PlanStepKind.SUMMARIZE,
            description="New step 2",
            status=PlanStepStatus.PENDING,
        ),
    ]
    
    plan.replace_remaining_steps(new_steps)
    assert len(plan.steps) == 3  # First step kept, 2 new steps added
    assert plan.steps[0].index == 0
    assert plan.steps[1].index == 1
    assert plan.steps[2].index == 2
    
    # Test advance to completion
    plan.advance()
    plan.advance()
    plan.advance()
    assert plan.status == PlanStatus.COMPLETED


def test_session_state_helpers():
    """Test SessionState helper methods."""
    session = SessionState(
        session_id="test-2",
        user_query="test",
    )
    
    # Test append_perception
    from orchestrated_agents.models import PerceptionSnapshot
    snapshot = PerceptionSnapshot(
        id="snap-1",
        turn_index=0,
        source="user",
        input_text="test",
    )
    session.append_perception(snapshot)
    assert len(session.perception_snapshots) == 1
    
    # Test append_retrieval
    from orchestrated_agents.models import RetrievalBundle
    bundle = RetrievalBundle(
        id="bundle-1",
        turn_index=0,
        query_used="test",
    )
    session.append_retrieval(bundle)
    assert len(session.retrieval_bundles) == 1
    
    # Test log_tool_perf
    record = ToolPerfRecord(
        tool_name="test_tool",
        success=True,
        latency_ms=100,
        step_id="step-1",
    )
    session.log_tool_perf(record)
    assert len(session.tool_performance.records) == 1


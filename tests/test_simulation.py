"""Tests for the simulation module."""

from orchestrated_agents.simulation import run_simulation


def test_run_simulation_returns_100_rows():
    results, tool_stats = run_simulation(num_tests=100)
    assert len(results) == 100
    assert len(tool_stats) > 0


def test_human_in_loop_flags_present():
    results, _ = run_simulation(num_tests=50)
    assert any(r.step_failed_human_in_loop for r in results)
    assert any(r.tool_failed_human_in_loop for r in results)


def test_tool_stats_consistent():
    _, tool_stats = run_simulation(num_tests=30)
    for stat in tool_stats:
        assert stat.total_calls == stat.success_count + stat.failure_count


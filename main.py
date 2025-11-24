"""Command-line entry point for orchestrated agents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from orchestrated_agents.coordinator import Coordinator
from orchestrated_agents.critic_agent import HeuristicCriticAgent
from orchestrated_agents.decision_agent import SimpleDecisionAgent
from orchestrated_agents.executor_adapter import StubExecutor
from orchestrated_agents.memory_agent import SimpleMemoryAgent
from orchestrated_agents.models import StrategyProfile
from orchestrated_agents.perception_agent import RuleBasedPerceptionAgent
from orchestrated_agents.retriever_agent import DummyRetrieverAgent
from orchestrated_agents.safe_executor import SafeExecutor
from orchestrated_agents.tool_registry import ToolRegistry


def build_default_coordinator() -> Coordinator:
    """Instantiate a coordinator with the default shallow tool stack."""
    tool_registry = ToolRegistry()
    tool_registry.register_simple(
        name="python",
        description="Default python tool",
        executor=lambda code: f"Executed python code: {code[:40]}...",
        input_schema={"code": {"type": "string"}},
        timeout_seconds=5.0,
        max_retries=3,
    )
    stub_executor = StubExecutor(
        responses={
            "python": "Python execution completed successfully",
            "calculator": "Calculation result: 42",
        }
    )
    executor = SafeExecutor(
        underlying_executor=stub_executor,
        tool_registry=tool_registry,
        default_timeout=5.0,
        max_retries=3,
    )
    perception = RuleBasedPerceptionAgent()
    retriever = DummyRetrieverAgent()
    memory = SimpleMemoryAgent()
    decision = SimpleDecisionAgent()
    critic = HeuristicCriticAgent()

    return Coordinator(
        perception_agent=perception,
        retriever_agent=retriever,
        memory_agent=memory,
        decision_agent=decision,
        critic_agent=critic,
        executor=executor,
        strategy=StrategyProfile.EXPLORATORY,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments for the orchestrated agents demo."""
    parser = argparse.ArgumentParser(description="Run the orchestrated agents pipeline.")
    parser.add_argument(
        "query",
        nargs="*",
        help="User query to run (if omitted, read from stdin).",
    )
    parser.add_argument(
        "--strategy",
        choices=[profile.value for profile in StrategyProfile],
        default=StrategyProfile.EXPLORATORY.value,
        help="Strategy profile to use for the run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for running the orchestrated agents coordinator."""
    args = parse_args(argv or sys.argv[1:])
    user_query = " ".join(args.query).strip()
    if not user_query:
        print("Enter your query (Ctrl+D to cancel):")
        user_query = sys.stdin.readline().strip()
        if not user_query:
            print("No query provided, exiting.")
            return 1

    coordinator = build_default_coordinator()
    coordinator.strategy = StrategyProfile(args.strategy)

    session = coordinator.run(user_query)

    print("\n=== Final Answer ===")
    print(session.final_answer or "No final answer generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

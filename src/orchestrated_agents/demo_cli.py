"""Demo CLI for orchestrated agents."""

import sys

from .coordinator import Coordinator
from .critic_agent import HeuristicCriticAgent
from .decision_agent import SimpleDecisionAgent
from .executor_adapter import StubExecutor
from .memory_agent import SimpleMemoryAgent
from .models import StrategyProfile
from .perception_agent import RuleBasedPerceptionAgent
from .retriever_agent import DummyRetrieverAgent


def main():
    """Main CLI entry point."""
    # Instantiate all agents with simple heuristic implementations
    perception_agent = RuleBasedPerceptionAgent()
    retriever_agent = DummyRetrieverAgent(
        knowledge_base={
            "python": "Python is a programming language",
            "calculator": "Calculator can perform arithmetic operations",
        }
    )
    memory_agent = SimpleMemoryAgent()
    decision_agent = SimpleDecisionAgent()
    critic_agent = HeuristicCriticAgent()
    executor = StubExecutor(
        responses={
            "python": "Python execution completed successfully",
            "calculator": "Calculation result: 42",
        }
    )
    
    # Create coordinator with chosen strategy
    strategy = StrategyProfile.CONSERVATIVE
    if len(sys.argv) > 1:
        strategy_str = sys.argv[1].lower()
        if strategy_str == "exploratory":
            strategy = StrategyProfile.EXPLORATORY
        elif strategy_str == "fallback":
            strategy = StrategyProfile.FALLBACK
    
    coordinator = Coordinator(
        perception_agent=perception_agent,
        retriever_agent=retriever_agent,
        memory_agent=memory_agent,
        decision_agent=decision_agent,
        critic_agent=critic_agent,
        executor=executor,
        strategy=strategy,
    )
    
    # Read user query
    if len(sys.argv) > 2:
        user_query = " ".join(sys.argv[2:])
    else:
        print("Enter your query (or 'exit' to quit):")
        user_query = input().strip()
        if not user_query or user_query.lower() == "exit":
            return
    
    # Run coordinator and print trace
    coordinator.run_and_print(user_query)


if __name__ == "__main__":
    main()


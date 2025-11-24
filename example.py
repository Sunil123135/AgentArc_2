"""Simple example of using orchestrated agents."""

from src.orchestrated_agents import Coordinator
from src.orchestrated_agents.models import StrategyProfile
from src.orchestrated_agents.perception_agent import RuleBasedPerceptionAgent
from src.orchestrated_agents.retriever_agent import DummyRetrieverAgent
from src.orchestrated_agents.memory_agent import SimpleMemoryAgent
from src.orchestrated_agents.decision_agent import SimpleDecisionAgent
from src.orchestrated_agents.critic_agent import HeuristicCriticAgent
from src.orchestrated_agents.executor_adapter import StubExecutor

def main():
    """Run a simple example."""
    # Create all agents
    perception_agent = RuleBasedPerceptionAgent()
    retriever_agent = DummyRetrieverAgent(
        knowledge_base={
            "python": "Python is a programming language for general-purpose programming",
            "calculator": "A calculator performs mathematical operations",
        }
    )
    memory_agent = SimpleMemoryAgent()
    decision_agent = SimpleDecisionAgent()
    critic_agent = HeuristicCriticAgent()
    executor = StubExecutor(
        responses={
            "python": "Execution completed: result = 4",
            "calculator": "Calculation result: 2 + 2 = 4",
        }
    )
    
    # Create coordinator
    coordinator = Coordinator(
        perception_agent=perception_agent,
        retriever_agent=retriever_agent,
        memory_agent=memory_agent,
        decision_agent=decision_agent,
        critic_agent=critic_agent,
        executor=executor,
        strategy=StrategyProfile.CONSERVATIVE,
    )
    
    # Run with a query
    print("Running coordinator with query: 'Calculate 2 + 2'")
    print("=" * 60)
    
    session = coordinator.run_and_print("Calculate 2 + 2")
    
    print("\n" + "=" * 60)
    print(f"Session completed: {session.done}")
    print(f"Final answer: {session.final_answer}")

if __name__ == "__main__":
    main()


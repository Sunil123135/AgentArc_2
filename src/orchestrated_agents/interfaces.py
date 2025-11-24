"""Abstract base classes for agents and executors."""

from abc import ABC, abstractmethod
from typing import Optional

from .models import (
    CriticReport,
    MemoryState,
    PerceptionSnapshot,
    PlanStep,
    PlanVersion,
    RetrievalBundle,
    SessionState,
    ToolPerfRecord,
)


class PerceptionAgent(ABC):
    """Abstract perception agent."""
    
    @abstractmethod
    def analyze_query(self, user_query: str, session: SessionState) -> PerceptionSnapshot:
        """Analyze a user query and return a perception snapshot."""
        pass
    
    @abstractmethod
    def analyze_step_result(
        self, step: PlanStep, raw_output: str, session: SessionState
    ) -> PerceptionSnapshot:
        """Analyze a step result and return a perception snapshot."""
        pass


class RetrieverAgent(ABC):
    """Abstract retriever agent."""
    
    @abstractmethod
    def retrieve(
        self,
        perception: PerceptionSnapshot,
        memory: MemoryState,
        session: SessionState,
    ) -> RetrievalBundle:
        """Retrieve relevant information based on perception."""
        pass


class MemoryAgent(ABC):
    """Abstract memory agent."""
    
    @abstractmethod
    def attach_relevant_memory(
        self, perception: PerceptionSnapshot, session: SessionState
    ) -> MemoryState:
        """Attach relevant memory to the session based on perception."""
        pass
    
    @abstractmethod
    def update_from_step(
        self,
        step: PlanStep,
        critic: Optional[CriticReport],
        success: bool,
        session: SessionState,
    ) -> MemoryState:
        """Update memory based on step execution."""
        pass


class DecisionAgent(ABC):
    """Abstract decision agent."""
    
    @abstractmethod
    def plan_initial(
        self,
        perception: PerceptionSnapshot,
        retrieval: Optional[RetrievalBundle],
        memory: MemoryState,
        session: SessionState,
    ) -> PlanVersion:
        """Create an initial plan based on perception and retrieval."""
        pass
    
    @abstractmethod
    def decide_next_step(
        self, plan: PlanVersion, session: SessionState
    ) -> Optional[PlanStep]:
        """Decide the next step to execute from the plan."""
        pass
    
    @abstractmethod
    def rewrite_plan(
        self,
        plan: PlanVersion,
        failed_step: PlanStep,
        critic: CriticReport,
        memory: MemoryState,
        session: SessionState,
    ) -> PlanVersion:
        """Rewrite a plan after a step failure."""
        pass


class CriticAgent(ABC):
    """Abstract critic agent."""
    
    @abstractmethod
    def review_result(
        self,
        step: PlanStep,
        perception: PerceptionSnapshot,
        retrieval: Optional[RetrievalBundle],
        session: SessionState,
    ) -> CriticReport:
        """Review a step result and return a critic report."""
        pass


class Executor(ABC):
    """Abstract executor."""
    
    @abstractmethod
    def execute_step(
        self, step: PlanStep, session: SessionState
    ) -> tuple[str, dict, ToolPerfRecord]:
        """Execute a plan step and return (result_text, result_payload, tool_perf_record)."""
        pass


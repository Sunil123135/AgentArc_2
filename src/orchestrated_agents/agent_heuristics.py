"""Input/output heuristics for agent validation."""

import re
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator

from .models import (
    CriticReport,
    MemoryState,
    PerceptionSnapshot,
    PlanStep,
    RetrievalBundle,
    SessionState,
)


class PerceptionHeuristics(BaseModel):
    """Heuristics for perception agent input/output validation."""
    
    min_query_length: int = Field(default=3, ge=1, le=1000)
    max_query_length: int = Field(default=5000, ge=100, le=50000)
    max_entities: int = Field(default=50, ge=1, le=200)
    max_sub_goals: int = Field(default=10, ge=1, le=50)
    
    def validate_query(self, query: str) -> tuple[bool, Optional[str]]:
        """Validate user query input."""
        if len(query) < self.min_query_length:
            return False, f"Query too short (minimum {self.min_query_length} characters)"
        if len(query) > self.max_query_length:
            return False, f"Query too long (maximum {self.max_query_length} characters)"
        
        # Check for suspicious patterns
        suspicious = [
            r'<script',
            r'javascript:',
            r'onerror=',
            r'eval\(',
        ]
        query_lower = query.lower()
        for pattern in suspicious:
            if re.search(pattern, query_lower):
                return False, f"Query contains suspicious pattern: {pattern}"
        
        return True, None
    
    def validate_perception_output(self, perception: PerceptionSnapshot) -> tuple[bool, Optional[str]]:
        """Validate perception agent output."""
        if len(perception.entities) > self.max_entities:
            return False, f"Too many entities (maximum {self.max_entities})"
        if len(perception.sub_goals) > self.max_sub_goals:
            return False, f"Too many sub-goals (maximum {self.max_sub_goals})"
        if not (0.0 <= perception.confidence <= 1.0):
            return False, "Confidence must be between 0.0 and 1.0"
        
        return True, None


class RetrieverHeuristics(BaseModel):
    """Heuristics for retriever agent input/output validation."""
    
    max_items: int = Field(default=20, ge=1, le=100)
    max_query_length: int = Field(default=500, ge=10, le=2000)
    min_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    
    def validate_retrieval_query(self, query: str) -> tuple[bool, Optional[str]]:
        """Validate retrieval query."""
        if len(query) < 3:
            return False, "Retrieval query too short"
        if len(query) > self.max_query_length:
            return False, f"Retrieval query too long (maximum {self.max_query_length} characters)"
        return True, None
    
    def validate_retrieval_output(self, bundle: RetrievalBundle) -> tuple[bool, Optional[str]]:
        """Validate retrieval bundle output."""
        if len(bundle.items) > self.max_items:
            return False, f"Too many retrieved items (maximum {self.max_items})"
        
        # Check relevance scores
        for item in bundle.items:
            if not (0.0 <= item.relevance <= 1.0):
                return False, f"Invalid relevance score: {item.relevance}"
        
        return True, None


class DecisionHeuristics(BaseModel):
    """Heuristics for decision agent input/output validation."""
    
    max_steps_per_plan: int = Field(default=20, ge=1, le=100)
    max_plan_rewrites: int = Field(default=5, ge=0, le=10)
    min_step_description_length: int = Field(default=5, ge=1, le=100)
    
    def validate_plan(self, plan_steps: list[PlanStep]) -> tuple[bool, Optional[str]]:
        """Validate plan structure."""
        if len(plan_steps) > self.max_steps_per_plan:
            return False, f"Plan has too many steps (maximum {self.max_steps_per_plan})"
        
        for step in plan_steps:
            if len(step.description) < self.min_step_description_length:
                return False, f"Step description too short (minimum {self.min_step_description_length} characters)"
        
        return True, None


class CriticHeuristics(BaseModel):
    """Heuristics for critic agent input/output validation."""
    
    min_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    max_issues: int = Field(default=20, ge=1, le=100)
    
    def validate_critic_output(self, report: CriticReport) -> tuple[bool, Optional[str]]:
        """Validate critic report output."""
        if not (0.0 <= report.quality_score <= 1.0):
            return False, "Quality score must be between 0.0 and 1.0"
        if not (0.0 <= report.hallucination_risk <= 1.0):
            return False, "Hallucination risk must be between 0.0 and 1.0"
        if len(report.issues) > self.max_issues:
            return False, f"Too many issues (maximum {self.max_issues})"
        
        return True, None


class MemoryHeuristics(BaseModel):
    """Heuristics for memory agent input/output validation."""
    
    max_short_term_items: int = Field(default=100, ge=1, le=1000)
    max_banned_tools: int = Field(default=50, ge=0, le=200)
    max_successful_tools: int = Field(default=100, ge=0, le=500)
    
    def validate_memory_output(self, memory: MemoryState) -> tuple[bool, Optional[str]]:
        """Validate memory state output."""
        if len(memory.short_term) > self.max_short_term_items:
            return False, f"Too many short-term memory items (maximum {self.max_short_term_items})"
        if len(memory.banned_tools) > self.max_banned_tools:
            return False, f"Too many banned tools (maximum {self.max_banned_tools})"
        if len(memory.successful_tools) > self.max_successful_tools:
            return False, f"Too many successful tool entries (maximum {self.max_successful_tools})"
        
        return True, None


class AgentHeuristics:
    """Container for all agent heuristics."""
    
    def __init__(self):
        """Initialize all heuristics."""
        self.perception = PerceptionHeuristics()
        self.retriever = RetrieverHeuristics()
        self.decision = DecisionHeuristics()
        self.critic = CriticHeuristics()
        self.memory = MemoryHeuristics()


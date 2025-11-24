"""Shared data models (blackboard) for the multi-agent system."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class StrategyProfile(str, Enum):
    """Strategy profiles for agent behavior."""
    CONSERVATIVE = "conservative"
    EXPLORATORY = "exploratory"
    FALLBACK = "fallback"


class PerceptionSnapshot(BaseModel):
    """What Perception Agent returns for either user query or step result."""
    
    id: str
    turn_index: int
    source: Literal["user", "step_result"]
    input_text: str
    entities: list[str] = Field(default_factory=list)
    intent: str = ""
    sub_goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    is_goal_satisfied: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    notes: str = ""


class RetrievedItem(BaseModel):
    """A single retrieved item."""
    
    source: str
    snippet: str
    url_or_id: str
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    summary: str = ""
    open_questions: list[str] = Field(default_factory=list)


class RetrievalBundle(BaseModel):
    """What Retriever Agent returns."""
    
    id: str
    turn_index: int
    query_used: str
    sources_used: list[Literal["web", "files", "vector_store", "memory"]] = Field(default_factory=list)
    items: list[RetrievedItem] = Field(default_factory=list)
    summary: str = ""
    open_questions: list[str] = Field(default_factory=list)


class PlanStepStatus(str, Enum):
    """Status of a plan step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStepKind(str, Enum):
    """Kind of plan step."""
    THINK = "think"
    RETRIEVE = "retrieve"
    EXECUTE = "execute"
    ASK_USER = "ask_user"
    SUMMARIZE = "summarize"


class PlanStep(BaseModel):
    """A single step in a plan."""
    
    id: str
    index: int
    kind: PlanStepKind
    description: str
    input_context: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    expected_outcome: str = ""
    result_text: Optional[str] = None
    result_payload: Optional[dict[str, Any]] = None
    status: PlanStepStatus = PlanStepStatus.PENDING
    attempts: int = 0
    critic_report_id: Optional[str] = None
    perception_snapshot_id: Optional[str] = None
    needs_plan_rewrite: bool = False
    notes: str = ""


class PlanStatus(str, Enum):
    """Status of a plan."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class PlanVersion(BaseModel):
    """A plan is a versioned sequence of steps."""
    
    id: str
    created_turn: int
    parent_plan_id: Optional[str] = None
    reason_for_creation: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    current_index: int = 0
    status: PlanStatus = PlanStatus.ACTIVE
    plan_text: str = ""
    
    def current_step(self) -> Optional[PlanStep]:
        """Get the current step."""
        if 0 <= self.current_index < len(self.steps):
            return self.steps[self.current_index]
        return None
    
    def advance(self) -> None:
        """Advance to the next step."""
        if self.current_index < len(self.steps):
            self.current_index += 1
        if self.current_index >= len(self.steps):
            self.status = PlanStatus.COMPLETED
    
    def replace_remaining_steps(self, new_steps: list[PlanStep]) -> None:
        """Replace steps from current_index onwards."""
        self.steps = self.steps[:self.current_index] + new_steps
        # Re-index the new steps
        for i, step in enumerate(new_steps):
            step.index = self.current_index + i


class CriticReport(BaseModel):
    """From Critic Agent."""
    
    id: str
    step_id: str
    turn_index: int
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    is_acceptable: bool = False
    issues: list[str] = Field(default_factory=list)
    hallucination_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    safety_flags: list[str] = Field(default_factory=list)
    rewrite_suggestion: Optional[str] = None
    requires_human_input: bool = False
    human_question: Optional[str] = None


class MemoryItemKind(str, Enum):
    """Kind of memory item."""
    FACT = "fact"
    PATTERN = "pattern"
    TOOL_FAILURE = "tool_failure"
    TOOL_SUCCESS = "tool_success"
    USER_PREF = "user_pref"


class MemoryItem(BaseModel):
    """A single memory item."""
    
    id: str
    kind: MemoryItemKind
    content: str
    tags: list[str] = Field(default_factory=list)
    created_turn: int


class MemoryState(BaseModel):
    """Short- and long-term memory, managed by Memory Agent."""
    
    short_term: list[MemoryItem] = Field(default_factory=list)
    long_term_refs: list[str] = Field(default_factory=list)
    banned_tools: set[str] = Field(default_factory=set)
    successful_tools: dict[str, int] = Field(default_factory=dict)
    notes: str = ""


class ToolPerfRecord(BaseModel):
    """A single tool performance record."""
    
    tool_name: str
    success: bool
    latency_ms: int
    step_id: str
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class ToolPerformanceLog(BaseModel):
    """Log of tool performance."""
    
    records: list[ToolPerfRecord] = Field(default_factory=list)

    def summarize_by_tool(self) -> dict[str, dict[str, float]]:
        """Aggregate performance statistics per tool."""
        summary: dict[str, dict[str, float]] = {}
        for record in self.records:
            tool_stats = summary.setdefault(
                record.tool_name,
                {
                    "success_count": 0,
                    "failure_count": 0,
                    "total_calls": 0,
                    "latency_sum": 0.0,
                },
            )
            tool_stats["total_calls"] += 1
            if record.success:
                tool_stats["success_count"] += 1
            else:
                tool_stats["failure_count"] += 1
            tool_stats["latency_sum"] += record.latency_ms
        
        for stats in summary.values():
            total = stats["total_calls"] or 1
            stats["avg_latency_ms"] = stats["latency_sum"] / total
            stats["success_rate"] = stats["success_count"] / total
            stats.pop("latency_sum", None)
        
        return summary

    def save_to_path(self, path: str | Path) -> Path:
        """Persist the log to disk as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return target

    @classmethod
    def load_from_path(cls, path: str | Path) -> "ToolPerformanceLog":
        """Load a tool performance log from disk."""
        source = Path(path)
        with source.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "records" in payload:
            payload["records"] = [
                ToolPerfRecord(**record) for record in payload["records"]
            ]
        return cls(**payload)


class SessionState(BaseModel):
    """Global state (blackboard) shared by all agents."""
    
    session_id: str
    user_query: str
    current_turn: int = 0
    strategy_profile: StrategyProfile = StrategyProfile.CONSERVATIVE
    perception_snapshots: list[PerceptionSnapshot] = Field(default_factory=list)
    retrieval_bundles: list[RetrievalBundle] = Field(default_factory=list)
    plans: list[PlanVersion] = Field(default_factory=list)
    active_plan_id: Optional[str] = None
    memory_state: MemoryState = Field(default_factory=MemoryState)
    tool_performance: ToolPerformanceLog = Field(default_factory=ToolPerformanceLog)
    final_answer: Optional[str] = None
    done: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)
    
    def get_active_plan(self) -> Optional[PlanVersion]:
        """Get the currently active plan."""
        if self.active_plan_id is None:
            return None
        for plan in self.plans:
            if plan.id == self.active_plan_id:
                return plan
        return None
    
    def set_active_plan(self, plan: PlanVersion) -> None:
        """Set the active plan."""
        if plan.id not in [p.id for p in self.plans]:
            self.plans.append(plan)
        self.active_plan_id = plan.id
    
    def append_perception(self, snapshot: PerceptionSnapshot) -> None:
        """Append a perception snapshot."""
        self.perception_snapshots.append(snapshot)
    
    def append_retrieval(self, bundle: RetrievalBundle) -> None:
        """Append a retrieval bundle."""
        self.retrieval_bundles.append(bundle)
    
    def log_tool_perf(self, record: ToolPerfRecord) -> None:
        """Log tool performance."""
        self.tool_performance.records.append(record)
    
    def mark_done(self, answer: str) -> None:
        """Mark the session as done with a final answer."""
        self.final_answer = answer
        self.done = True
        if self.active_plan_id:
            plan = self.get_active_plan()
            if plan:
                plan.status = PlanStatus.COMPLETED


"""Memory Agent - manages short and long-term memory."""

import uuid
from typing import Callable, Optional

from .interfaces import MemoryAgent
from .models import (
    CriticReport,
    MemoryItem,
    MemoryItemKind,
    MemoryState,
    PerceptionSnapshot,
    PlanStep,
    SessionState,
)
from .agent_prompts import AgentPrompt, resolve_prompt
from .prompt_runtime import PromptRuntimeMixin

# Threshold for banning tools
FAILURE_THRESHOLD = 3


class SimpleMemoryAgent(PromptRuntimeMixin, MemoryAgent):
    """Simple memory agent implementation."""
    
    def __init__(
        self,
        llm_completion: Optional[Callable[[str], str]] = None,
        prompt: Optional[AgentPrompt | str] = None,
    ):
        """Initialize with optional LLM completion function."""
        self.llm_completion = llm_completion
        self.prompt = resolve_prompt("memory", prompt)
    
    def get_prompt(self) -> AgentPrompt:
        """Expose the active prompt for this agent."""
        return self.prompt
    
    def attach_relevant_memory(
        self, perception: PerceptionSnapshot, session: SessionState
    ) -> MemoryState:
        """Attach relevant memory to the session based on perception."""
        memory = session.memory_state
        
        # Scan existing memory for entity overlap
        relevant_items = []
        for mem_item in memory.short_term:
            if self._is_relevant(mem_item, perception):
                relevant_items.append(mem_item)
        
        # Add notes about relevant memory
        if relevant_items:
            notes_parts = [f"Found {len(relevant_items)} relevant memory items:"]
            for item in relevant_items[:3]:  # Top 3
                notes_parts.append(f"- {item.kind.value}: {item.content[:50]}")
            memory.notes = "\n".join(notes_parts)
        else:
            memory.notes = "No relevant memory found for this perception."
        
        memory.notes = self._append_guidance(
            memory.notes,
            self._prompt_guidance("attach_relevant_memory", perception.input_text),
        )
        
        return memory
    
    def update_from_step(
        self,
        step: PlanStep,
        critic: Optional[CriticReport],
        success: bool,
        session: SessionState,
    ) -> MemoryState:
        """Update memory based on step execution."""
        memory = session.memory_state
        turn = session.current_turn
        
        # Create memory item based on step outcome
        if step.tool_name:
            if success:
                # Record tool success
                item = MemoryItem(
                    id=str(uuid.uuid4()),
                    kind=MemoryItemKind.TOOL_SUCCESS,
                    content=f"Tool {step.tool_name} succeeded: {step.description}",
                    tags=[step.tool_name, "success"],
                    created_turn=turn,
                )
                memory.short_term.append(item)
                
                # Update success count
                memory.successful_tools[step.tool_name] = (
                    memory.successful_tools.get(step.tool_name, 0) + 1
                )
            else:
                # Record tool failure
                item = MemoryItem(
                    id=str(uuid.uuid4()),
                    kind=MemoryItemKind.TOOL_FAILURE,
                    content=f"Tool {step.tool_name} failed: {step.description}. Error: {step.notes}",
                    tags=[step.tool_name, "failure"],
                    created_turn=turn,
                )
                memory.short_term.append(item)
                
                # Check if tool should be banned
                failure_count = sum(
                    1
                    for m in memory.short_term
                    if m.kind == MemoryItemKind.TOOL_FAILURE
                    and step.tool_name in m.tags
                )
                if failure_count >= FAILURE_THRESHOLD:
                    memory.banned_tools.add(step.tool_name)
                    memory.notes = f"Tool {step.tool_name} banned after {failure_count} failures."
        
        # Store patterns from critic if available
        if critic and critic.issues:
            pattern_item = MemoryItem(
                id=str(uuid.uuid4()),
                kind=MemoryItemKind.PATTERN,
                content=f"Pattern detected: {', '.join(critic.issues)}",
                tags=["pattern", "critic"],
                created_turn=turn,
            )
            memory.short_term.append(pattern_item)
        
        memory = self._update_tool_performance_notes(session, memory)
        memory.notes = self._append_guidance(
            memory.notes,
            self._prompt_guidance("memory_update_from_step", step.description),
        )
        
        return memory
    
    def suggest_banned_or_preferred_tools(
        self, session: SessionState
    ) -> MemoryState:
        """Update banned_tools and successful_tools based on history."""
        memory = session.memory_state
        
        # Count failures per tool
        tool_failures: dict[str, int] = {}
        for mem_item in memory.short_term:
            if mem_item.kind == MemoryItemKind.TOOL_FAILURE:
                for tag in mem_item.tags:
                    if tag != "failure":
                        tool_failures[tag] = tool_failures.get(tag, 0) + 1
        
        # Ban tools that exceed threshold
        for tool_name, count in tool_failures.items():
            if count >= FAILURE_THRESHOLD:
                memory.banned_tools.add(tool_name)
        
        return memory
    
    def _is_relevant(self, mem_item: MemoryItem, perception: PerceptionSnapshot) -> bool:
        """Check if memory item is relevant to perception."""
        # Check tag overlap
        if any(tag.lower() in [e.lower() for e in perception.entities] for tag in mem_item.tags):
            return True
        
        # Check content overlap
        content_lower = mem_item.content.lower()
        for entity in perception.entities:
            if entity.lower() in content_lower:
                return True
        
        return False

    def _update_tool_performance_notes(
        self, session: SessionState, memory: MemoryState
    ) -> MemoryState:
        """Append aggregated tool performance notes into memory."""
        summary = session.tool_performance.summarize_by_tool()
        if not summary:
            return memory
        
        lines = ["Tool performance summary:"]
        for tool_name in sorted(summary.keys()):
            stats = summary[tool_name]
            lines.append(
                f"- {tool_name}: {stats['success_count']}/{stats['total_calls']} success, "
                f"{stats['avg_latency_ms']:.1f}ms avg latency"
            )
        
        addition = "\n".join(lines)
        if memory.notes:
            memory.notes = f"{memory.notes}\n\n{addition}"
        else:
            memory.notes = addition
        
        return memory


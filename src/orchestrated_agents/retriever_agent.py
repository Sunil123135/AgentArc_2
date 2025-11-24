"""Retriever Agent - retrieves relevant information."""

import uuid
from typing import Callable, Optional

from .interfaces import RetrieverAgent
from .models import MemoryState, PerceptionSnapshot, RetrievedItem, RetrievalBundle, SessionState
from .agent_prompts import AgentPrompt, resolve_prompt
from .prompt_runtime import PromptRuntimeMixin


class DummyRetrieverAgent(PromptRuntimeMixin, RetrieverAgent):
    """Dummy retriever agent that uses memory and simple in-memory knowledge base."""
    
    def __init__(
        self,
        llm_completion: Optional[Callable[[str], str]] = None,
        knowledge_base: Optional[dict[str, str]] = None,
        prompt: Optional[AgentPrompt | str] = None,
    ):
        """Initialize with optional LLM completion and knowledge base."""
        self.llm_completion = llm_completion
        self.knowledge_base = knowledge_base or {}
        self.prompt = resolve_prompt("retriever", prompt)

    def get_prompt(self) -> AgentPrompt:
        """Expose the active prompt for this agent."""
        return self.prompt
    
    def retrieve(
        self,
        perception: PerceptionSnapshot,
        memory: MemoryState,
        session: SessionState,
    ) -> RetrievalBundle:
        """Retrieve relevant information."""
        bundle_id = str(uuid.uuid4())
        items: list[RetrievedItem] = []
        
        # Search memory for relevant items
        for mem_item in memory.short_term:
            if self._is_relevant(mem_item, perception):
                items.append(
                    RetrievedItem(
                        source="memory",
                        snippet=mem_item.content,
                        url_or_id=mem_item.id,
                        relevance=0.7,
                        summary=mem_item.content[:100],
                        open_questions=[],
                    )
                )
        
        # Search knowledge base
        for key, value in self.knowledge_base.items():
            if self._matches_perception(key, value, perception):
                items.append(
                    RetrievedItem(
                        source="files",
                        snippet=value,
                        url_or_id=key,
                        relevance=0.6,
                        summary=value[:100],
                        open_questions=[],
                    )
                )
        
        # Build summary
        summary = self._build_summary(items)
        summary = self._append_guidance(
            summary,
            self._prompt_guidance("retrieve_context", perception.input_text),
        )
        
        # Generate open questions
        open_questions = self._generate_open_questions(perception, items)
        guidance_question = self._prompt_guidance(
            "generate_follow_up_question", perception.input_text
        )
        if guidance_question:
            open_questions.append(guidance_question.strip())
        
        return RetrievalBundle(
            id=bundle_id,
            turn_index=session.current_turn,
            query_used=perception.input_text,
            sources_used=["memory", "files"] if items else [],
            items=items,
            summary=summary,
            open_questions=open_questions,
        )
    
    def _is_relevant(self, mem_item, perception: PerceptionSnapshot) -> bool:
        """Check if memory item is relevant to perception."""
        # Check tag overlap
        if any(tag in perception.entities for tag in mem_item.tags):
            return True
        
        # Check content overlap
        content_lower = mem_item.content.lower()
        for entity in perception.entities:
            if entity.lower() in content_lower:
                return True
        
        # Check intent/sub-goal overlap
        for sub_goal in perception.sub_goals:
            if any(word in content_lower for word in sub_goal.lower().split()):
                return True
        
        return False
    
    def _matches_perception(
        self, key: str, value: str, perception: PerceptionSnapshot
    ) -> bool:
        """Check if knowledge base entry matches perception."""
        text_to_search = f"{key} {value}".lower()
        
        # Check entity overlap
        for entity in perception.entities:
            if entity.lower() in text_to_search:
                return True
        
        # Check sub-goal overlap
        for sub_goal in perception.sub_goals:
            if any(word in text_to_search for word in sub_goal.lower().split()):
                return True
        
        return False
    
    def _build_summary(self, items: list[RetrievedItem]) -> str:
        """Build a summary from retrieved items."""
        if not items:
            return "No relevant information found."
        
        snippets = [item.snippet[:200] for item in items[:3]]  # Top 3 items
        return " | ".join(snippets)
    
    def _generate_open_questions(
        self, perception: PerceptionSnapshot, items: list[RetrievedItem]
    ) -> list[str]:
        """Generate open questions based on perception and retrieved items."""
        questions = []
        
        # If no items found, suggest what might be needed
        if not items:
            if perception.uncertainties:
                questions.append("What additional context is needed to resolve uncertainties?")
            if perception.sub_goals:
                questions.append(f"How should we approach: {perception.sub_goals[0]}?")
        
        # If items found but uncertainties exist
        if items and perception.uncertainties:
            questions.append("How can we verify the retrieved information?")
        
        return questions


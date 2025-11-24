"""Perception Agent - analyzes queries and step results."""

import re
import uuid
from typing import Callable, Optional

from .interfaces import PerceptionAgent
from .models import PerceptionSnapshot, PlanStep, SessionState
from .agent_prompts import AgentPrompt, resolve_prompt
from .prompt_runtime import PromptRuntimeMixin


class RuleBasedPerceptionAgent(PromptRuntimeMixin, PerceptionAgent):
    """Rule-based perception agent using regex and keyword matching."""
    
    def __init__(
        self,
        llm_completion: Optional[Callable[[str], str]] = None,
        prompt: Optional[AgentPrompt | str] = None,
    ):
        """Initialize with optional LLM completion function and prompt."""
        self.llm_completion = llm_completion
        self.prompt = resolve_prompt("perception", prompt)
    
    def get_prompt(self) -> AgentPrompt:
        """Expose the active prompt for this agent."""
        return self.prompt
    
    def analyze_query(self, user_query: str, session: SessionState) -> PerceptionSnapshot:
        """Analyze a user query."""
        snapshot_id = str(uuid.uuid4())
        
        # Extract entities (capitalized words, numbers, quoted strings)
        entities = self._extract_entities(user_query)
        
        # Classify intent
        intent = self._classify_intent(user_query)
        
        # Extract constraints
        constraints = self._extract_constraints(user_query)
        
        # Extract sub-goals (split by "and", "then", "also")
        sub_goals = self._extract_sub_goals(user_query)
        
        # Detect uncertainties (question words, modals)
        uncertainties = self._detect_uncertainties(user_query)
        
        # Check if goal seems satisfied (contains "done", "complete", etc.)
        is_goal_satisfied = self._check_goal_satisfied(user_query)
        
        # Calculate confidence based on clarity
        confidence = self._calculate_confidence(user_query, entities, intent)
        
        notes = f"Analyzed query with {len(entities)} entities"
        notes = self._append_guidance(
            notes,
            self._prompt_guidance("analyze_user_query", user_query),
        )
        return PerceptionSnapshot(
            id=snapshot_id,
            turn_index=session.current_turn,
            source="user",
            input_text=user_query,
            entities=entities,
            intent=intent,
            sub_goals=sub_goals,
            constraints=constraints,
            uncertainties=uncertainties,
            is_goal_satisfied=is_goal_satisfied,
            confidence=confidence,
            notes=notes,
        )
    
    def analyze_step_result(
        self, step: PlanStep, raw_output: str, session: SessionState
    ) -> PerceptionSnapshot:
        """Analyze a step result."""
        snapshot_id = str(uuid.uuid4())
        
        # Extract entities from output
        entities = self._extract_entities(raw_output)
        
        # Check if expected outcome keywords appear in output
        expected_keywords = self._extract_keywords(step.expected_outcome)
        output_keywords = self._extract_keywords(raw_output.lower())
        overlap = len(set(expected_keywords) & set(output_keywords))
        is_goal_satisfied = overlap > 0 and len(expected_keywords) > 0
        
        # Classify intent of the result
        intent = "result" if step.kind == "execute" else "information"
        
        # Extract constraints from output
        constraints = self._extract_constraints(raw_output)
        
        # Detect uncertainties
        uncertainties = self._detect_uncertainties(raw_output)
        
        # Calculate confidence based on expected outcome match
        confidence = (
            overlap / len(expected_keywords) if expected_keywords else 0.5
        )
        
        notes = f"Analyzed step {step.id} result"
        notes = self._append_guidance(
            notes,
            self._prompt_guidance("analyze_step_result", raw_output),
        )
        return PerceptionSnapshot(
            id=snapshot_id,
            turn_index=session.current_turn,
            source="step_result",
            input_text=raw_output,
            entities=entities,
            intent=intent,
            sub_goals=[],
            constraints=constraints,
            uncertainties=uncertainties,
            is_goal_satisfied=is_goal_satisfied,
            confidence=confidence,
            notes=notes,
        )
    
    def _extract_entities(self, text: str) -> list[str]:
        """Extract entities from text."""
        entities = []
        # Capitalized words (potential proper nouns)
        entities.extend(re.findall(r'\b[A-Z][a-z]+\b', text))
        # Numbers
        entities.extend(re.findall(r'\b\d+\b', text))
        # Quoted strings
        entities.extend(re.findall(r'"([^"]+)"', text))
        entities.extend(re.findall(r"'([^']+)'", text))
        return list(set(entities))
    
    def _classify_intent(self, text: str) -> str:
        """Classify intent of the query."""
        text_lower = text.lower()
        if any(word in text_lower for word in ["what", "how", "why", "when", "where", "?"]):
            return "question"
        elif any(word in text_lower for word in ["calculate", "compute", "solve", "run"]):
            return "command"
        elif any(word in text_lower for word in ["analyze", "compare", "evaluate"]):
            return "analysis"
        elif any(word in text_lower for word in ["create", "make", "generate", "build"]):
            return "creation"
        else:
            return "general"
    
    def _extract_constraints(self, text: str) -> list[str]:
        """Extract constraints from text."""
        constraints = []
        text_lower = text.lower()
        
        # Time constraints
        if any(word in text_lower for word in ["before", "by", "deadline", "eod", "asap"]):
            constraints.append("time_constraint")
        
        # Safety constraints
        if any(word in text_lower for word in ["safe", "secure", "careful"]):
            constraints.append("safety_constraint")
        
        # Tool constraints
        if any(word in text_lower for word in ["without", "avoid", "don't use"]):
            constraints.append("tool_constraint")
        
        return constraints
    
    def _extract_sub_goals(self, text: str) -> list[str]:
        """Extract sub-goals from text."""
        # Split by common conjunctions
        parts = re.split(r'\s+(?:and|then|also|,)\s+', text, flags=re.IGNORECASE)
        return [part.strip() for part in parts if part.strip() and len(part.strip()) > 3]
    
    def _detect_uncertainties(self, text: str) -> list[str]:
        """Detect uncertainties in text."""
        uncertainties = []
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["maybe", "perhaps", "might", "could", "uncertain"]):
            uncertainties.append("modal_uncertainty")
        if "?" in text:
            uncertainties.append("question_uncertainty")
        if any(word in text_lower for word in ["not sure", "unclear", "ambiguous"]):
            uncertainties.append("clarity_uncertainty")
        
        return uncertainties
    
    def _check_goal_satisfied(self, text: str) -> bool:
        """Check if goal appears to be satisfied."""
        text_lower = text.lower()
        return any(word in text_lower for word in ["done", "complete", "finished", "solved"])
    
    def _calculate_confidence(
        self, text: str, entities: list[str], intent: str
    ) -> float:
        """Calculate confidence score."""
        confidence = 0.5  # Base confidence
        
        # More entities = higher confidence
        if len(entities) > 0:
            confidence += min(0.2, len(entities) * 0.05)
        
        # Clear intent = higher confidence
        if intent != "general":
            confidence += 0.1
        
        # Longer text = potentially more context
        if len(text) > 50:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text (simple word splitting)."""
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        return [w for w in words if w not in stop_words and len(w) > 2]


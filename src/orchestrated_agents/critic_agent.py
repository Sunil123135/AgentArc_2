"""Critic Agent - reviews step results."""

import uuid
from typing import Callable, Optional

from .interfaces import CriticAgent
from .models import (
    CriticReport,
    PerceptionSnapshot,
    PlanStep,
    RetrievalBundle,
    SessionState,
)
from .agent_prompts import AgentPrompt, resolve_prompt
from .prompt_runtime import PromptRuntimeMixin


class HeuristicCriticAgent(PromptRuntimeMixin, CriticAgent):
    """Heuristic-based critic agent."""
    
    def __init__(
        self,
        llm_completion: Optional[Callable[[str], str]] = None,
        prompt: Optional[AgentPrompt | str] = None,
    ):
        """Initialize with optional LLM completion function."""
        self.llm_completion = llm_completion
        self.prompt = resolve_prompt("critic", prompt)
        # Banned words that trigger safety checks
        self.safety_banned_words = [
            "delete", "remove", "destroy", "kill", "harm", "dangerous",
            "illegal", "unauthorized", "hack", "exploit",
        ]
    
    def get_prompt(self) -> AgentPrompt:
        """Expose the active prompt for this agent."""
        return self.prompt
    
    def review_result(
        self,
        step: PlanStep,
        perception: PerceptionSnapshot,
        retrieval: Optional[RetrievalBundle],
        session: SessionState,
    ) -> CriticReport:
        """Review a step result and return a critic report."""
        report_id = str(uuid.uuid4())
        
        # Check if result matches expected outcome
        expected_keywords = self._extract_keywords(step.expected_outcome.lower())
        result_keywords = self._extract_keywords(
            (step.result_text or "").lower()
        )
        
        overlap = len(set(expected_keywords) & set(result_keywords))
        expected_len = len(expected_keywords)
        match_ratio = overlap / expected_len if expected_len > 0 else 0.0
        
        # Calculate quality score
        quality_score = match_ratio
        
        # Check against retrieval if available
        if retrieval and step.result_text:
            retrieval_overlap = self._check_retrieval_overlap(
                step.result_text, retrieval
            )
            quality_score = (quality_score + retrieval_overlap) / 2
        
        # Detect hallucination risk
        hallucination_risk = self._detect_hallucination_risk(
            step, perception, retrieval
        )
        
        # Safety checks
        safety_flags = self._check_safety(step.result_text or "")
        
        # Determine if acceptable
        is_acceptable = (
            quality_score >= 0.5
            and hallucination_risk < 0.7
            and len(safety_flags) == 0
            and step.status.value == "success"
        )
        
        # Generate issues
        issues = []
        if quality_score < 0.5:
            issues.append("Low quality: result doesn't match expected outcome")
        if hallucination_risk > 0.7:
            issues.append("High hallucination risk: result contains unverified information")
        if safety_flags:
            issues.append(f"Safety concerns: {', '.join(safety_flags)}")
        if step.status.value == "failed":
            issues.append("Step execution failed")
        
        guidance = self._prompt_guidance("critic_review", step.result_text or step.description)
        if guidance:
            issues.append(f"LLM guidance: {guidance}")
        
        # Determine if rewrite is needed
        needs_rewrite = not is_acceptable and step.status.value == "failed"
        rewrite_suggestion = None
        if needs_rewrite:
            rewrite_suggestion = (
                f"Retry step with alternative approach. "
                f"Issues: {', '.join(issues)}"
            )
        
        # Check if human input is required
        requires_human_input = len(safety_flags) > 0 or hallucination_risk > 0.8
        human_question = None
        if requires_human_input:
            if safety_flags:
                human_question = f"Safety concern detected: {', '.join(safety_flags)}. Proceed?"
            elif hallucination_risk > 0.8:
                human_question = "High uncertainty in result. Verify before proceeding?"
        
        return CriticReport(
            id=report_id,
            step_id=step.id,
            turn_index=session.current_turn,
            quality_score=quality_score,
            is_acceptable=is_acceptable,
            issues=issues,
            hallucination_risk=hallucination_risk,
            safety_flags=safety_flags,
            rewrite_suggestion=rewrite_suggestion,
            requires_human_input=requires_human_input,
            human_question=human_question,
        )
    
    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text."""
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "is", "are", "was", "were", "be",
            "been", "have", "has", "had", "do", "does", "did",
        }
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def _check_retrieval_overlap(
        self, result_text: str, retrieval: RetrievalBundle
    ) -> float:
        """Check overlap between result and retrieved information."""
        if not retrieval.items:
            return 0.5  # Neutral if no retrieval
        
        result_keywords = set(self._extract_keywords(result_text.lower()))
        
        # Get keywords from all retrieved items
        retrieval_keywords = set()
        for item in retrieval.items:
            retrieval_keywords.update(self._extract_keywords(item.snippet.lower()))
            retrieval_keywords.update(self._extract_keywords(item.summary.lower()))
        
        if not retrieval_keywords:
            return 0.5
        
        overlap = len(result_keywords & retrieval_keywords)
        total_unique = len(result_keywords | retrieval_keywords)
        
        return overlap / total_unique if total_unique > 0 else 0.0
    
    def _detect_hallucination_risk(
        self,
        step: PlanStep,
        perception: PerceptionSnapshot,
        retrieval: Optional[RetrievalBundle],
    ) -> float:
        """Detect hallucination risk in the result."""
        if not step.result_text:
            return 0.0
        
        risk = 0.0
        
        # Extract entities from result
        result_entities = self._extract_entities(step.result_text)
        
        # Check if result contains new entities not in input or retrieval
        input_entities = set(e.lower() for e in perception.entities)
        result_entities_lower = set(e.lower() for e in result_entities)
        
        new_entities = result_entities_lower - input_entities
        
        # If retrieval available, check against it
        if retrieval:
            retrieval_entities = set()
            for item in retrieval.items:
                retrieval_entities.update(self._extract_entities(item.snippet))
            retrieval_entities_lower = set(e.lower() for e in retrieval_entities)
            new_entities = new_entities - retrieval_entities_lower
        
        # More new entities = higher risk
        if new_entities:
            risk += min(0.5, len(new_entities) * 0.1)
        
        # If no retrieval and result is long, higher risk
        if not retrieval and len(step.result_text) > 200:
            risk += 0.2
        
        return min(1.0, risk)
    
    def _extract_entities(self, text: str) -> list[str]:
        """Extract entities from text (simple version)."""
        import re
        entities = []
        # Capitalized words
        entities.extend(re.findall(r'\b[A-Z][a-z]+\b', text))
        return list(set(entities))
    
    def _check_safety(self, text: str) -> list[str]:
        """Check for safety issues in text."""
        if not text:
            return []
        
        text_lower = text.lower()
        flags = []
        
        for banned_word in self.safety_banned_words:
            if banned_word in text_lower:
                flags.append(f"contains_{banned_word}")
        
        return flags


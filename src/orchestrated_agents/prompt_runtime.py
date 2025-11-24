"""Utilities for applying agent prompts at runtime."""

from __future__ import annotations

from typing import Callable, Optional

from .agent_prompts import AgentPrompt, run_prompted_completion


class PromptRuntimeMixin:
    """Mixin that wires AgentPrompt data into optional LLM calls."""

    llm_completion: Optional[Callable[[str], str]]
    prompt: AgentPrompt

    def _prompt_guidance(self, task: str, content: str) -> Optional[str]:
        """Invoke the provided LLM with the agent's prompt, if available."""
        return run_prompted_completion(self.llm_completion, self.prompt, task, content)

    @staticmethod
    def _append_guidance(base_text: str, guidance: Optional[str]) -> str:
        """Append LLM guidance to a block of text if guidance exists."""
        if not guidance:
            return base_text
        base_text = base_text.strip()
        if base_text:
            return f"{base_text}\nLLM guidance: {guidance}"
        return f"LLM guidance: {guidance}"


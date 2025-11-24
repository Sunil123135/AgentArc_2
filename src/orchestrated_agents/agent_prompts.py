"""Centralized prompt definitions loaded from JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict


@dataclass(frozen=True)
class AgentPrompt:
    """Structured metadata for an agent prompt."""

    name: str
    role: str
    instructions: str


PROMPT_FILES: Dict[str, str] = {
    "perception": "perception.json",
    "retriever": "retriever.json",
    "memory": "memory.json",
    "decision": "decision.json",
    "critic": "critic.json",
    "executor": "executor.json",
}

PROMPT_DIR = Path(__file__).with_name("prompts")


@lru_cache(maxsize=None)
def _load_prompt_from_file(key: str) -> AgentPrompt:
    """Load a prompt from its JSON definition."""
    filename = PROMPT_FILES[key]
    path = PROMPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found for key '{key}': {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return AgentPrompt(**data)


DEFAULT_AGENT_PROMPTS: Dict[str, AgentPrompt] = {
    key: _load_prompt_from_file(key) for key in PROMPT_FILES
}


def list_default_prompts() -> Dict[str, AgentPrompt]:
    """Expose a shallow copy of the default prompt mapping."""
    return dict(DEFAULT_AGENT_PROMPTS)


def resolve_prompt(key: str, prompt: AgentPrompt | str | None) -> AgentPrompt:
    """Return an AgentPrompt instance for the requested agent/component."""
    base = DEFAULT_AGENT_PROMPTS[key]
    if prompt is None:
        return base
    if isinstance(prompt, AgentPrompt):
        return prompt
    return AgentPrompt(name=base.name, role=base.role, instructions=str(prompt))


def render_llm_prompt(agent_prompt: AgentPrompt, task: str, content: str) -> str:
    """Combine structured instructions with runtime content."""
    return (
        f"{agent_prompt.name} ({agent_prompt.role})\n"
        f"Instructions:\n{agent_prompt.instructions}\n\n"
        f"Task: {task}\n"
        f"Input:\n{content}\n"
        "Respond with concise guidance."
    )


def run_prompted_completion(
    llm_completion: Callable[[str], str] | None,
    agent_prompt: AgentPrompt,
    task: str,
    content: str,
) -> str | None:
    """Invoke the provided LLM callable with prompt context if available."""
    if llm_completion is None:
        return None
    prompt_text = render_llm_prompt(agent_prompt, task, content)
    try:
        return llm_completion(prompt_text)
    except Exception:
        return None


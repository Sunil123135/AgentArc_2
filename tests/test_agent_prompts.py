"""Tests for agent prompt infrastructure."""

from orchestrated_agents.agent_prompts import (
    AgentPrompt,
    DEFAULT_AGENT_PROMPTS,
    list_default_prompts,
    resolve_prompt,
)


def test_default_prompt_keys_present():
    """All primary agents expose default prompts."""
    expected = {"perception", "retriever", "memory", "decision", "critic", "executor"}
    assert expected.issubset(DEFAULT_AGENT_PROMPTS.keys())


def test_resolve_prompt_overrides_text():
    """resolve_prompt should wrap simple strings into AgentPrompt."""
    custom = resolve_prompt("decision", "Custom decision instructions")
    assert isinstance(custom, AgentPrompt)
    assert custom.instructions == "Custom decision instructions"
    assert custom.name == DEFAULT_AGENT_PROMPTS["decision"].name


def test_list_default_prompts_returns_copy():
    """list_default_prompts must return a shallow copy to avoid mutation."""
    prompts_a = list_default_prompts()
    prompts_b = list_default_prompts()
    prompts_a["perception"] = AgentPrompt(
        name="Override",
        role="Override",
        instructions="Override",
    )
    assert prompts_b["perception"] == DEFAULT_AGENT_PROMPTS["perception"]


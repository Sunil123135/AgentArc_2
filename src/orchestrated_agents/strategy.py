"""Strategy profiles and utilities."""

from .models import StrategyProfile


def default_limits(strategy: StrategyProfile) -> dict[str, int]:
    """Return default limits for a strategy profile."""
    if strategy == StrategyProfile.CONSERVATIVE:
        return {
            "max_steps": 5,
            "max_retries": 2,
            "max_plan_rewrites": 1,
        }
    elif strategy == StrategyProfile.EXPLORATORY:
        return {
            "max_steps": 15,
            "max_retries": 3,
            "max_plan_rewrites": 3,
        }
    else:  # FALLBACK
        return {
            "max_steps": 10,
            "max_retries": 4,
            "max_plan_rewrites": 2,
        }


def should_do_parallel_retrieval(strategy: StrategyProfile) -> bool:
    """Determine if parallel retrieval should be used."""
    return strategy == StrategyProfile.EXPLORATORY


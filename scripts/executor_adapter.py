from __future__ import annotations

"""
Executor adapter with simulated tool failures (MCP-style).

- When imported as part of the `orchestrated_agents` package, it uses normal package imports.
- When run directly (e.g., `python src/orchestrated_agents/executor_adapter.py`),
  it patches sys.path so `orchestrated_agents` becomes importable.
"""

import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple, Any

# ---------------------------------------------------------------------
# Make orchestrated_agents importable when running this file directly
# ---------------------------------------------------------------------
if __name__ == "__main__" and __package__ is None:
    # File is at: <PROJECT_ROOT>/src/orchestrated_agents/executor_adapter.py
    project_root = Path(__file__).resolve().parents[2]  # go up from src/orchestrated_agents/...
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

# ---------------------------------------------------------------------
# Now safe to import from orchestrated_agents
# ---------------------------------------------------------------------
from orchestrated_agents.interfaces import Executor
from orchestrated_agents.models import ToolPerfRecord, PlanStep, SessionState


# ---------------------------------------------------------------------
# Custom exception to trigger Human-In-Loop escalation
# ---------------------------------------------------------------------
class ToolFailureError(Exception):
    """
    Exception raised when a tool fails (simulated MCP outage).

    The Coordinator checks the `hil_category` attribute to decide
    whether to invoke the tool-failure Human-In-Loop path.
    """

    def __init__(self, message: str, hil_category: str = "tool_failure") -> None:
        super().__init__(message)
        self.hil_category = hil_category


# ---------------------------------------------------------------------
# Executor Implementation With Forced Failures
# ---------------------------------------------------------------------
class FailingExecutor(Executor):
    """
    Executor that simulates persistent tool failures for selected tool names.

    Any step whose `tool_name` is in `failing_tools` will raise `ToolFailureError`,
    which Coordinator will treat as a tool failure and call `_handle_tool_failure`.
    """

    def __init__(self, failing_tools: Optional[Iterable[str]] = None) -> None:
        self.failing_tools: Set[str] = set(failing_tools or [])

    def execute_step(
        self,
        step: PlanStep,
        session: SessionState,
    ) -> Tuple[str, dict[str, Any], ToolPerfRecord]:

        tool_name = step.tool_name or "unknown"
        start_ns = time.time()

        # Simulate failure if tool matches failing list
        if tool_name in self.failing_tools:
            # IMPORTANT: message format lines up with Coordinator._handle_tool_failure:
            # "Tool {tool_name} failed with error '{error_message}'. ..."
            error_message = f"Tool {tool_name} failed due to simulated MCP outage."
            raise ToolFailureError(error_message, hil_category="tool_failure")

        # Otherwise simulate a successful execution
        result_text = f"Simulated successful execution for tool '{tool_name}'."
        result_payload: dict[str, Any] = {
            "tool": tool_name,
            "ok": True,
        }
        latency_ms = int((time.time() - start_ns) * 1000)

        perf = ToolPerfRecord(
            tool_name=tool_name,
            success=True,
            latency_ms=latency_ms,
            step_id=step.id,
        )
        return result_text, result_payload, perf


# ---------------------------------------------------------------------
# Optional: simple debug run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print(
        "[executor_adapter] Module loaded successfully.\n"
        "This file is normally imported, not used as an entry point.\n"
        "FailingExecutor is ready for use in Coordinator with failing_tools={'python'} or {'calculator'}."
    )

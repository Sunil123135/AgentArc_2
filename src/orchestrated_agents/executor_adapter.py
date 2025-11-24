"""Executor adapter - wraps existing MCP/Executor code."""

from datetime import datetime
from typing import Any, Optional

from .interfaces import Executor
from .models import PlanStep, PlanStepKind, SessionState, ToolPerfRecord


class StubExecutor(Executor):
    """Stub executor for testing - always succeeds with predictable output."""
    
    def __init__(self, responses: Optional[dict[str, str]] = None):
        """Initialize with optional response mapping."""
        self.responses = responses or {}
        self.call_count: dict[str, int] = {}
    
    def execute_step(
        self, step: PlanStep, session: SessionState
    ) -> tuple[str, dict[str, Any], ToolPerfRecord]:
        """Execute a step and return result."""
        if step.kind != PlanStepKind.EXECUTE:
            # Non-execute steps return dummy result
            return (
                "Step completed",
                {},
                ToolPerfRecord(
                    tool_name="none",
                    success=True,
                    latency_ms=10,
                    step_id=step.id,
                ),
            )
        
        tool_name = step.tool_name or "unknown"
        self.call_count[tool_name] = self.call_count.get(tool_name, 0) + 1
        
        # Check if we have a custom response
        if tool_name in self.responses:
            result_text = self.responses[tool_name]
        else:
            result_text = f"Executed {tool_name} with args {step.tool_args}"
        
        result_payload = {
            "tool": tool_name,
            "args": step.tool_args,
            "call_count": self.call_count[tool_name],
        }
        
        record = ToolPerfRecord(
            tool_name=tool_name,
            success=True,
            latency_ms=50,
            step_id=step.id,
        )
        
        return result_text, result_payload, record


class FailingExecutor(Executor):
    """Executor that fails for specific tools on first attempt."""
    
    def __init__(self, failing_tools: Optional[set[str]] = None):
        """Initialize with set of tools that should fail."""
        self.failing_tools = failing_tools or set()
        self.attempts: dict[str, int] = {}
    
    def execute_step(
        self, step: PlanStep, session: SessionState
    ) -> tuple[str, dict[str, Any], ToolPerfRecord]:
        """Execute a step, failing for specified tools."""
        if step.kind != PlanStepKind.EXECUTE:
            return (
                "Step completed",
                {},
                ToolPerfRecord(
                    tool_name="none",
                    success=True,
                    latency_ms=10,
                    step_id=step.id,
                ),
            )
        
        tool_name = step.tool_name or "unknown"
        attempt_key = f"{step.id}_{tool_name}"
        self.attempts[attempt_key] = self.attempts.get(attempt_key, 0) + 1
        
        # Fail on first attempt if tool is in failing set
        if tool_name in self.failing_tools and self.attempts[attempt_key] == 1:
            raise Exception(f"Tool {tool_name} failed on first attempt")
        
        # Succeed on retry or if not in failing set
        result_text = f"Executed {tool_name} successfully (attempt {self.attempts[attempt_key]})"
        result_payload = {
            "tool": tool_name,
            "args": step.tool_args,
            "attempt": self.attempts[attempt_key],
        }
        
        record = ToolPerfRecord(
            tool_name=tool_name,
            success=True,
            latency_ms=50,
            step_id=step.id,
        )
        
        return result_text, result_payload, record


class MCPExecutor(Executor):
    """Adapter for existing MCP/Executor code."""
    
    def __init__(self, underlying_executor: Any):
        """Initialize with underlying executor instance."""
        self.executor = underlying_executor
    
    def execute_step(
        self, step: PlanStep, session: SessionState
    ) -> tuple[str, dict[str, Any], ToolPerfRecord]:
        """Execute a step using the underlying executor."""
        if step.kind != PlanStepKind.EXECUTE:
            return (
                "Step completed",
                {},
                ToolPerfRecord(
                    tool_name="none",
                    success=True,
                    latency_ms=10,
                    step_id=step.id,
                ),
            )
        
        tool_name = step.tool_name or "unknown"
        start_time = datetime.now()
        
        try:
            # Delegate to underlying executor
            # Assuming it has a method like execute(tool_name, args)
            if hasattr(self.executor, "execute"):
                result = self.executor.execute(tool_name, step.tool_args or {})
                result_text = str(result)
                result_payload = {"result": result}
            elif hasattr(self.executor, "run"):
                result = self.executor.run(tool_name, step.tool_args or {})
                result_text = str(result)
                result_payload = {"result": result}
            else:
                # Fallback: assume executor is callable
                result = self.executor(tool_name, step.tool_args or {})
                result_text = str(result)
                result_payload = {"result": result}
            
            end_time = datetime.now()
            latency_ms = int((end_time - start_time).total_seconds() * 1000)
            
            record = ToolPerfRecord(
                tool_name=tool_name,
                success=True,
                latency_ms=latency_ms,
                step_id=step.id,
            )
            
            return result_text, result_payload, record
            
        except Exception as e:
            end_time = datetime.now()
            latency_ms = int((end_time - start_time).total_seconds() * 1000)
            
            record = ToolPerfRecord(
                tool_name=tool_name,
                success=False,
                latency_ms=latency_ms,
                step_id=step.id,
                error_message=str(e),
            )
            
            raise


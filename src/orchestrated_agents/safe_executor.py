"""Safe executor with timeout, retry, and validation."""

import signal
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, Optional

from .executor_adapter import Executor
from .models import PlanStep, SessionState, ToolPerfRecord
from .tool_registry import ToolRegistry
from .agent_prompts import AgentPrompt, resolve_prompt, run_prompted_completion


class SafeExecutor(Executor):
    """Executor wrapper that adds safety checks, timeouts, and retries."""
    
    def __init__(
        self,
        underlying_executor: Executor,
        tool_registry: ToolRegistry,
        default_timeout: float = 5.0,
        max_retries: int = 3,
        prompt: Optional[AgentPrompt | str] = None,
        llm_completion: Optional[Callable[[str], str]] = None,
    ):
        """Initialize safe executor.
        
        Args:
            underlying_executor: The actual executor to wrap
            tool_registry: Registry for tool validation
            default_timeout: Default timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.underlying_executor = underlying_executor
        self.tool_registry = tool_registry
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.prompt = resolve_prompt("executor", prompt)
        self.llm_completion = llm_completion
    
    def execute_step(
        self, step: PlanStep, session: SessionState
    ) -> tuple[str, dict[str, Any], ToolPerfRecord]:
        """Execute a step with safety checks, timeout, and retries."""
        # Skip non-execute steps
        if step.kind.value != "execute":
            return self.underlying_executor.execute_step(step, session)
        
        tool_name = step.tool_name or "unknown"
        
        # Check if tool is registered
        if not self.tool_registry.is_registered(tool_name):
            error_msg = f"Tool '{tool_name}' is not registered in tool registry"
            return self._create_error_result(step, error_msg)
        
        # Get tool schema
        tool_schema = self.tool_registry.get_tool(tool_name)
        if not tool_schema:
            error_msg = f"Tool '{tool_name}' schema not found"
            return self._create_error_result(step, error_msg)
        
        # Validate input safety
        input_data = step.tool_args or {}
        is_safe, safety_error = self.tool_registry.check_safety(tool_name, input_data)
        if not is_safe:
            return self._create_error_result(step, f"Safety check failed: {safety_error}")
        
        # Validate input schema
        is_valid, validation_error = self.tool_registry.validate_input(tool_name, input_data)
        if not is_valid:
            return self._create_error_result(step, f"Input validation failed: {validation_error}")
        
        # Execute with retries
        timeout = tool_schema.timeout_seconds or self.default_timeout
        max_attempts = min(tool_schema.max_retries or self.max_retries, self.max_retries)
        
        last_error = None
        for attempt in range(max_attempts):
            try:
                # Execute with timeout
                result_text, result_payload, tool_perf = self._execute_with_timeout(
                    step, session, timeout
                )
                
                # Validate output
                if result_payload:
                    is_valid_output, output_error = self.tool_registry.validate_output(
                        tool_name, result_payload
                    )
                    if not is_valid_output:
                        # Output validation failed, but we got a result
                        # Log warning but don't fail
                        result_text = f"{result_text} [Output validation warning: {output_error}]"
                
                return result_text, result_payload, tool_perf
                
            except TimeoutError as e:
                last_error = f"Timeout after {timeout}s (attempt {attempt + 1}/{max_attempts})"
                if attempt < max_attempts - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
            except Exception as e:
                last_error = str(e)
                if attempt < max_attempts - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
        
        # All retries exhausted
        return self._create_error_result(step, f"Execution failed after {max_attempts} attempts: {last_error}")
    
    def _execute_with_timeout(
        self,
        step: PlanStep,
        session: SessionState,
        timeout: float,
    ) -> tuple[str, dict[str, Any], ToolPerfRecord]:
        """Execute step with timeout using thread pool."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.underlying_executor.execute_step, step, session)
            try:
                result = future.result(timeout=timeout)
                return result
            except FutureTimeoutError:
                future.cancel()
                raise TimeoutError(f"Step execution timed out after {timeout}s")
    
    def _create_error_result(
        self, step: PlanStep, error_msg: str
    ) -> tuple[str, dict[str, Any], ToolPerfRecord]:
        """Create an error result."""
        guidance = run_prompted_completion(
            self.llm_completion,
            self.prompt,
            task="executor_error",
            content=f"Tool: {step.tool_name}\nError: {error_msg}",
        )
        payload: dict[str, Any] = {"error": error_msg}
        if guidance:
            payload["llm_guidance"] = guidance
        return (
            f"Error: {error_msg}",
            payload,
            ToolPerfRecord(
                tool_name=step.tool_name or "unknown",
                success=False,
                latency_ms=0,
                step_id=step.id,
                error_message=error_msg,
            ),
        )


"""Decision Agent - creates and manages plans."""

import uuid
from typing import Callable, Optional

from .interfaces import DecisionAgent
from .models import (
    CriticReport,
    MemoryState,
    PerceptionSnapshot,
    PlanStep,
    PlanStepKind,
    PlanStatus,
    PlanStepStatus,
    PlanVersion,
    RetrievalBundle,
    SessionState,
    StrategyProfile,
)
from .strategy import default_limits
from .agent_prompts import AgentPrompt, resolve_prompt
from .prompt_runtime import PromptRuntimeMixin


class SimpleDecisionAgent(PromptRuntimeMixin, DecisionAgent):
    """Simple decision agent implementation."""
    
    def __init__(
        self,
        llm_completion: Optional[Callable[[str], str]] = None,
        prompt: Optional[AgentPrompt | str] = None,
    ):
        """Initialize with optional LLM completion function."""
        self.llm_completion = llm_completion
        self.prompt = resolve_prompt("decision", prompt)
    
    def get_prompt(self) -> AgentPrompt:
        """Expose the active prompt for this agent."""
        return self.prompt
    
    def plan_initial(
        self,
        perception: PerceptionSnapshot,
        retrieval: Optional[RetrievalBundle],
        memory: MemoryState,
        session: SessionState,
    ) -> PlanVersion:
        """Create an initial plan based on perception and retrieval."""
        plan_id = str(uuid.uuid4())
        steps: list[PlanStep] = []
        step_index = 0
        
        # Determine max steps based on strategy
        limits = default_limits(session.strategy_profile)
        max_steps = limits["max_steps"]
        
        # Step 1: Optional retrieve step if context is missing
        if not retrieval or len(retrieval.items) == 0:
            if session.strategy_profile != StrategyProfile.CONSERVATIVE:
                steps.append(
                    PlanStep(
                        id=str(uuid.uuid4()),
                        index=step_index,
                        kind=PlanStepKind.RETRIEVE,
                        description="Retrieve relevant information",
                        input_context=perception.input_text,
                        expected_outcome="Relevant information retrieved",
                        status=PlanStepStatus.PENDING,
                    )
                )
                step_index += 1
        
        # Step 2: Execute steps based on sub-goals or intent
        if perception.sub_goals:
            for sub_goal in perception.sub_goals[:max_steps - len(steps) - 1]:
                tool_name = self._choose_tool_for_intent(
                    perception.intent, memory, task_context=sub_goal
                )
                steps.append(
                    PlanStep(
                        id=str(uuid.uuid4()),
                        index=step_index,
                        kind=PlanStepKind.EXECUTE,
                        description=f"Execute: {sub_goal}",
                        input_context=sub_goal,
                        tool_name=tool_name,
                        tool_args={},
                        expected_outcome=f"Completed: {sub_goal}",
                        status=PlanStepStatus.PENDING,
                    )
                )
                step_index += 1
        else:
            # Single execute step based on intent
            tool_name = self._choose_tool_for_intent(
                perception.intent, memory, task_context=perception.input_text
            )
            steps.append(
                PlanStep(
                    id=str(uuid.uuid4()),
                    index=step_index,
                    kind=PlanStepKind.EXECUTE,
                    description=f"Execute: {perception.input_text}",
                    input_context=perception.input_text,
                    tool_name=tool_name,
                    tool_args={},
                    expected_outcome="Task completed successfully",
                    status=PlanStepStatus.PENDING,
                )
            )
            step_index += 1
        
        # Final step: Summarize
        steps.append(
            PlanStep(
                id=str(uuid.uuid4()),
                index=step_index,
                kind=PlanStepKind.SUMMARIZE,
                description="Summarize results and produce final answer",
                input_context="All previous step results",
                expected_outcome="Final answer produced",
                status=PlanStepStatus.PENDING,
            )
        )
        
        plan_text = f"Plan to {perception.intent}: {len(steps)} steps"
        plan_text = self._append_guidance(
            plan_text,
            self._prompt_guidance("plan_initial", perception.input_text),
        )
        
        return PlanVersion(
            id=plan_id,
            created_turn=session.current_turn,
            reason_for_creation="initial",
            steps=steps,
            current_index=0,
            status=PlanStatus.ACTIVE,
            plan_text=plan_text,
        )
    
    def decide_next_step(
        self, plan: PlanVersion, session: SessionState
    ) -> Optional[PlanStep]:
        """Decide the next step to execute from the plan."""
        current = plan.current_step()
        
        # If there's a current step and it's not completed, return it
        if current and current.status not in [PlanStepStatus.SUCCESS, PlanStepStatus.SKIPPED]:
            return current
        
        # If no remaining steps and no final answer, create a summarize step
        if plan.current_index >= len(plan.steps) and session.final_answer is None:
            # This shouldn't happen if plan is managed correctly, but handle it
            return None
        
        return None
    
    def rewrite_plan(
        self,
        plan: PlanVersion,
        failed_step: PlanStep,
        critic: CriticReport,
        memory: MemoryState,
        session: SessionState,
    ) -> PlanVersion:
        """Rewrite a plan after a step failure."""
        new_plan_id = str(uuid.uuid4())
        new_steps: list[PlanStep] = []
        step_index = failed_step.index + 1
        
        # Keep all steps up to and including the failed step (mark as failed)
        # Actually, we keep steps before the failed one, and replace from failed onward
        kept_steps = plan.steps[:failed_step.index]
        
        # Determine max steps based on strategy
        limits = default_limits(session.strategy_profile)
        remaining_steps = limits["max_steps"] - len(kept_steps)
        
        # If the failed step was a retrieve, try a different approach
        if failed_step.kind == PlanStepKind.RETRIEVE:
            # Try alternative retrieve or skip to execute
            if session.strategy_profile != StrategyProfile.CONSERVATIVE:
                new_steps.append(
                    PlanStep(
                        id=str(uuid.uuid4()),
                        index=step_index,
                        kind=PlanStepKind.RETRIEVE,
                        description="Alternative retrieval approach",
                        input_context=failed_step.input_context,
                        expected_outcome="Relevant information retrieved via alternative method",
                        status=PlanStepStatus.PENDING,
                    )
                )
                step_index += 1
        
        # If the failed step was an execute, try alternative tool
        elif failed_step.kind == PlanStepKind.EXECUTE:
            # Choose alternative tool (not banned)
            alternative_tool = self._choose_alternative_tool(
                failed_step.tool_name, memory
            )
            
            new_steps.append(
                PlanStep(
                    id=str(uuid.uuid4()),
                    index=step_index,
                    kind=PlanStepKind.EXECUTE,
                    description=f"Retry with alternative approach: {failed_step.description}",
                    input_context=failed_step.input_context,
                    tool_name=alternative_tool,
                    tool_args=failed_step.tool_args or {},
                    expected_outcome=failed_step.expected_outcome,
                    status=PlanStepStatus.PENDING,
                )
            )
            step_index += 1
        
        # Add summarize step if we have room
        if remaining_steps > len(new_steps):
            new_steps.append(
                PlanStep(
                    id=str(uuid.uuid4()),
                    index=step_index,
                    kind=PlanStepKind.SUMMARIZE,
                    description="Summarize results and produce final answer",
                    input_context="All previous step results",
                    expected_outcome="Final answer produced",
                    status=PlanStepStatus.PENDING,
                )
            )
        
        # Create new plan version
        plan_text = f"Rewritten plan after failure: {len(new_steps)} new steps"
        plan_text = self._append_guidance(
            plan_text,
            self._prompt_guidance("rewrite_plan", failed_step.description),
        )
        
        new_plan = PlanVersion(
            id=new_plan_id,
            created_turn=session.current_turn,
            parent_plan_id=plan.id,
            reason_for_creation=f"rewrite_after_failure of step {failed_step.id}",
            steps=kept_steps + new_steps,
            current_index=len(kept_steps),  # Continue from where we left off
            status=PlanStatus.ACTIVE,
            plan_text=plan_text,
        )
        
        return new_plan
    
    def _choose_tool_for_intent(
        self, intent: str, memory: MemoryState, task_context: Optional[str] = None
    ) -> str:
        """Choose a tool based on intent, avoiding banned tools."""
        # Simple mapping
        tool_map = {
            "question": "python",
            "command": "python",
            "analysis": "python",
            "creation": "python",
        }
        tool = tool_map.get(intent, "python")

        if task_context:
            context = task_context.lower()
            if any(keyword in context for keyword in ["calculate", "sum", "math", "add"]):
                tool = "calculator"
            elif any(keyword in context for keyword in ["search", "web", "lookup", "internet"]):
                tool = "web_search"
            elif any(keyword in context for keyword in ["script", "python", "code"]):
                tool = "python"
        
        # If tool is banned, use fallback
        if tool in memory.banned_tools:
            return self._choose_alternative_tool(tool, memory)
        
        return tool
    
    def _choose_alternative_tool(
        self, original_tool: Optional[str], memory: MemoryState
    ) -> str:
        """Choose an alternative tool, avoiding banned tools."""
        alternatives = ["python", "calculator", "web_search"]
        
        # Remove original and banned tools
        available = [
            t for t in alternatives
            if t != original_tool and t not in memory.banned_tools
        ]
        
        return available[0] if available else "python"  # Last resort


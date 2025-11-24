"""Coordinator - main orchestration loop."""

import uuid
from pathlib import Path
from typing import Any, Callable, Optional
import inspect

from .interfaces import (
    CriticAgent,
    DecisionAgent,
    Executor,
    MemoryAgent,
    PerceptionAgent,
    RetrieverAgent,
)
from .models import (
    CriticReport,
    PlanStep,
    PlanStepKind,
    PlanStepStatus,
    SessionState,
    StrategyProfile,
)
from .strategy import default_limits


class Coordinator:
    """Main coordinator that orchestrates the multi-agent loop."""
    
    def __init__(
        self,
        perception_agent: PerceptionAgent,
        retriever_agent: RetrieverAgent,
        memory_agent: MemoryAgent,
        decision_agent: DecisionAgent,
        critic_agent: CriticAgent,
        executor: Executor,
        strategy: StrategyProfile = StrategyProfile.CONSERVATIVE,
        tool_log_dir: Optional[str | Path] = "tool_logs",
        user_input_callback: Optional[Callable[[str], str]] = None,
        human_input_callback: Optional[Callable[[str], str]] = None,
    ):
        """Initialize coordinator with all agents and executor."""
        self.perception_agent = perception_agent
        self.retriever_agent = retriever_agent
        self.memory_agent = memory_agent
        self.decision_agent = decision_agent
        self.critic_agent = critic_agent
        self.executor = executor
        self.strategy = strategy
        self.tool_log_dir = Path(tool_log_dir) if tool_log_dir else None
        self.user_input_callback = user_input_callback or (lambda q: "User response")
        self.human_input_callback = human_input_callback or (lambda q: "Proceed")
        self._human_input_accepts_session = False
        try:
            sig = inspect.signature(self.human_input_callback)
            self._human_input_accepts_session = len(sig.parameters) >= 2
        except (TypeError, ValueError):
            self._human_input_accepts_session = False
    
    def run(self, user_query: str, metadata: Optional[dict[str, Any]] = None) -> SessionState:
        """Run the main orchestration loop."""
        # Initialize SessionState
        session = SessionState(
            session_id=str(uuid.uuid4()),
            user_query=user_query,
            current_turn=0,
            strategy_profile=self.strategy,
        )
        if metadata:
            session.meta.update(metadata)
        
        # Step 1: Analyze user query
        perception = self.perception_agent.analyze_query(user_query, session)
        session.append_perception(perception)
        session.current_turn += 1
        
        # Step 2: Attach relevant memory
        memory = self.memory_agent.attach_relevant_memory(perception, session)
        session.memory_state = memory
        
        # Step 3: Retrieve if needed (based on perception or strategy)
        retrieval = None
        if not perception.is_goal_satisfied:
            retrieval = self.retriever_agent.retrieve(perception, memory, session)
            session.append_retrieval(retrieval)
        
        # Step 4: Create initial plan
        plan = self.decision_agent.plan_initial(
            perception, retrieval, memory, session
        )
        session.set_active_plan(plan)
        
        # Get limits from strategy
        limits = default_limits(self.strategy)
        max_steps = limits["max_steps"]
        max_plan_rewrites = limits["max_plan_rewrites"]
        plan_rewrite_count = 0
        
        # Main loop
        step_count = 0
        while not session.done and step_count < max_steps:
            step_count += 1
            
            # Get active plan
            plan = session.get_active_plan()
            if not plan:
                break
            
            # Get next step
            step = self.decision_agent.decide_next_step(plan, session)
            if not step:
                # No more steps, try to finalize
                if session.final_answer is None:
                    # Create a summarize step if needed
                    self._create_final_summary(session)
                break
            
            # Handle ask_user steps
            if step.kind == PlanStepKind.ASK_USER:
                user_response = self.user_input_callback(step.description)
                # Create perception snapshot from user response
                user_perception = self.perception_agent.analyze_query(
                    user_response, session
                )
                session.append_perception(user_perception)
                step.result_text = user_response
                step.status = PlanStepStatus.SUCCESS
                plan.advance()
                continue
            
            # Handle summarize steps
            if step.kind == PlanStepKind.SUMMARIZE:
                # Create final summary from all step results
                plan = session.get_active_plan()
                if plan:
                    results = []
                    for s in plan.steps:
                        if s.status == PlanStepStatus.SUCCESS and s.result_text and s.kind != PlanStepKind.SUMMARIZE:
                            results.append(s.result_text)
                    
                    if results:
                        step.result_text = "Final answer produced: " + " ".join(results)
                    else:
                        step.result_text = "Final answer produced: Task completed"
                    
                    step.status = PlanStepStatus.SUCCESS
                    # Create a dummy tool perf record for summarize steps
                    from .models import ToolPerfRecord
                    tool_perf = ToolPerfRecord(
                        tool_name="none",
                        success=True,
                        latency_ms=10,
                        step_id=step.id,
                    )
                    session.log_tool_perf(tool_perf)
                    # Continue to critic review, then mark as done
                else:
                    # No plan, skip execution
                    continue
            
            # Execute step (skip for summarize as it's handled above)
            if step.kind != PlanStepKind.SUMMARIZE:
                step.status = PlanStepStatus.RUNNING
                try:
                    result_text, result_payload, tool_perf = self.executor.execute_step(
                        step, session
                    )
                    step.result_text = result_text
                    step.result_payload = result_payload
                    step.status = PlanStepStatus.SUCCESS
                    
                    # Log tool performance
                    session.log_tool_perf(tool_perf)
                
                except Exception as e:
                    step.status = PlanStepStatus.FAILED
                    step.notes = str(e)
                    step.attempts += 1
                    # Create tool perf record for failure
                    from .models import ToolPerfRecord
                    tool_perf = ToolPerfRecord(
                        tool_name=step.tool_name or "unknown",
                        success=False,
                        latency_ms=0,
                        step_id=step.id,
                        error_message=str(e),
                    )
                    session.log_tool_perf(tool_perf)
                    hil_category = getattr(e, "hil_category", None)
                    if hil_category == "tool_failure":
                        self._handle_tool_failure(session, step, str(e))
                    elif hil_category == "step_failure":
                        self._handle_step_failure(session, step, str(e))
            
            # Analyze step result
            result_perception = self.perception_agent.analyze_step_result(
                step, step.result_text or "", session
            )
            session.append_perception(result_perception)
            step.perception_snapshot_id = result_perception.id
            
            # Get latest retrieval for critic
            latest_retrieval = (
                session.retrieval_bundles[-1] if session.retrieval_bundles else None
            )
            
            # Review result with critic
            critic_report = self.critic_agent.review_result(
                step, result_perception, latest_retrieval, session
            )
            step.critic_report_id = critic_report.id
            
            # Update memory
            memory = self.memory_agent.update_from_step(
                step, critic_report, step.status == PlanStepStatus.SUCCESS, session
            )
            session.memory_state = memory
            
            # Handle critic feedback
            if critic_report.requires_human_input:
                human_response = self.human_input_callback(
                    critic_report.human_question or "Proceed?"
                )
                # For now, assume proceed if response is positive
                if "no" not in human_response.lower() and "stop" not in human_response.lower():
                    step.status = PlanStepStatus.SUCCESS
                else:
                    step.status = PlanStepStatus.FAILED
                    session.mark_done("User requested stop.")
                    break
            
            if critic_report.is_acceptable:
                step.status = PlanStepStatus.SUCCESS
                plan.advance()
                
                # If this was a summarize step, mark session as done
                if step.kind == PlanStepKind.SUMMARIZE and session.final_answer is None:
                    session.mark_done(step.result_text or "Task completed")
                    break
            else:
                step.status = PlanStepStatus.FAILED
                step.needs_plan_rewrite = True
                if session.meta.get("is_complex_query"):
                    self._handle_step_failure(session, step, "Complex query step failed")
                
                # Rewrite plan if allowed
                if plan_rewrite_count < max_plan_rewrites:
                    plan_rewrite_count += 1
                    new_plan = self.decision_agent.rewrite_plan(
                        plan, step, critic_report, memory, session
                    )
                    session.set_active_plan(new_plan)
                else:
                    # Max rewrites reached, mark as done with error
                    failure_reason = (
                        f"Plan failed after {max_plan_rewrites} rewrites. "
                        f"Last error: {self._summarize_critic_issues(critic_report, step)}"
                    )
                    final_message = self._trigger_human_in_loop(
                        session,
                        reason=failure_reason,
                        failed_step=step,
                        critic_report=critic_report,
                    )
                    session.mark_done(final_message)
                    break
            
            # Check if plan is completed
            if plan.status.value == "completed":
                if session.final_answer is None:
                    self._create_final_summary(session)
                break
        
        # Finalize if not done
        if not session.done and session.final_answer is None:
            if step_count >= max_steps:
                reason = f"Reached maximum steps ({max_steps}) without completion."
                final_message = self._trigger_human_in_loop(session, reason=reason)
                session.mark_done(final_message)
            else:
                session.mark_done("Plan completed but no final answer generated.")
        
        self._persist_tool_performance_log(session)
        return session
    
    def _create_final_summary(self, session: SessionState) -> None:
        """Create a final summary from all step results."""
        plan = session.get_active_plan()
        if not plan:
            session.mark_done("No plan available for summarization.")
            return
        
        # Collect all successful step results
        results = []
        for step in plan.steps:
            if step.status == PlanStepStatus.SUCCESS and step.result_text:
                results.append(f"Step {step.index}: {step.result_text}")
        
        if results:
            final_answer = "\n".join(results)
        else:
            final_answer = "Task completed but no results to summarize."
        
        session.mark_done(final_answer)
    
    def _trigger_human_in_loop(
        self,
        session: SessionState,
        reason: str,
        failed_step: Optional[PlanStep] = None,
        critic_report: Optional[CriticReport] = None,
        category: str = "plan_failure",
    ) -> str:
        """Record a Human-In-Loop escalation and return the composed message."""
        suggestion = self._suggest_follow_up_plan(session, failed_step, critic_report)
        prompt = (
            f"Plan escalation triggered for reason: {reason}. "
            f"Suggest next action for the agent."
        )
        human_response = self._call_human_input(prompt, session)
        escalation_record = {
            "reason": reason,
            "suggested_plan": suggestion,
            "turn": session.current_turn,
            "failed_step_id": failed_step.id if failed_step else None,
            "critic_report_id": critic_report.id if critic_report else None,
            "agent_listening": True,
            "category": category,
        }
        session.meta.setdefault("human_in_loop_escalations", []).append(escalation_record)
        session.meta["agent_listening_for_human"] = True
        self._record_hil_event(
            session,
            category=category,
            prompt=prompt,
            response=human_response,
            step_id=failed_step.id if failed_step else None,
        )
        
        return (
            f"{reason} Human-In-Loop escalation required. "
            f"Suggested plan: {suggestion} Agent listens for human guidance."
        )

    def _handle_tool_failure(
        self, session: SessionState, step: PlanStep, error_message: str
    ) -> None:
        """Handle human-in-loop flow for tool failures."""
        prompt = (
            f"Tool {step.tool_name or 'unknown'} failed with error '{error_message}'. "
            "Provide manual guidance or fallback suggestion."
        )
        response = self._call_human_input(prompt, session)
        session.meta["tool_failed_human_in_loop"] = True
        self._record_hil_event(
            session,
            category="tool_failure",
            prompt=prompt,
            response=response,
            step_id=step.id,
        )
        if response:
            notes = step.notes or ""
            step.notes = f"{notes} Human input: {response}".strip()

    def _handle_step_failure(
        self, session: SessionState, step: PlanStep, reason: str
    ) -> None:
        """Handle human-in-loop flow for complex step failures."""
        prompt = (
            f"Step '{step.description}' failed ({reason}). "
            "Provide manual advice to unblock the plan."
        )
        response = self._call_human_input(prompt, session)
        session.meta["step_failed_human_in_loop"] = True
        self._record_hil_event(
            session,
            category="step_failure",
            prompt=prompt,
            response=response,
            step_id=step.id,
        )
        if response:
            notes = step.notes or ""
            step.notes = f"{notes} Human input: {response}".strip()

    def _call_human_input(self, prompt: str, session: SessionState) -> str:
        """Call the configured human input callback, optionally passing session."""
        if self.human_input_callback is None:
            return "Proceed"
        try:
            if self._human_input_accepts_session:
                return self.human_input_callback(prompt, session)
            return self.human_input_callback(prompt)
        except TypeError:
            return self.human_input_callback(prompt)

    def _record_hil_event(
        self,
        session: SessionState,
        category: str,
        prompt: str,
        response: Optional[str],
        step_id: Optional[str],
    ) -> None:
        """Record a structured human-in-loop event for reporting."""
        session.meta.setdefault("human_in_loop_events", []).append(
            {
                "category": category,
                "prompt": prompt,
                "response": response,
                "step_id": step_id,
                "turn": session.current_turn,
            }
        )
    
    def _suggest_follow_up_plan(
        self,
        session: SessionState,
        failed_step: Optional[PlanStep],
        critic_report: Optional[CriticReport],
    ) -> str:
        """Create a concise follow-up plan suggestion for human review."""
        if critic_report and critic_report.rewrite_suggestion:
            return critic_report.rewrite_suggestion
        if failed_step:
            tool = failed_step.tool_name or "manual review"
            return (
                f"Review failed step '{failed_step.description}' and assist with tool '{tool}'."
            )
        plan = session.get_active_plan()
        if plan:
            return f"Review plan '{plan.plan_text}' and outline manual next steps."
        return "Review recent context and craft a manual recovery plan."
    
    def _summarize_critic_issues(
        self, critic_report: CriticReport, failed_step: PlanStep
    ) -> str:
        """Summarize critic issues or fallback to step notes."""
        if critic_report.issues:
            return ", ".join(critic_report.issues)
        if failed_step.notes:
            return failed_step.notes
        return "Unspecified agent error"
    
    def _persist_tool_performance_log(self, session: SessionState) -> None:
        """Save tool performance log to disk and note location in session metadata."""
        if not self.tool_log_dir:
            return
        
        self.tool_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.tool_log_dir / f"{session.session_id}_tool_perf.json"
        session.tool_performance.save_to_path(log_path)
        session.meta["tool_performance_log_path"] = str(log_path)
    
    def run_and_print(self, user_query: str) -> SessionState:
        """Run coordinator and print a trace of steps."""
        print(f"=== Starting orchestration for query: {user_query} ===\n")
        
        session = self.run(user_query)
        
        print(f"\n=== Final Answer ===")
        print(session.final_answer or "No final answer generated.")
        
        print(f"\n=== Step Trace ===")
        for plan in session.plans:
            print(f"\nPlan {plan.id} ({plan.status.value}): {plan.plan_text}")
            for step in plan.steps:
                status_icon = {
                    "pending": "[P]",
                    "running": "[R]",
                    "success": "[OK]",
                    "failed": "[X]",
                    "skipped": "[S]",
                }.get(step.status.value, "[?]")
                
                print(
                    f"  {status_icon} Step {step.index} [{step.kind.value}]: "
                    f"{step.description}"
                )
                if step.result_text:
                    result_preview = step.result_text[:100]
                    print(f"      Result: {result_preview}...")
                if step.notes:
                    print(f"      Notes: {step.notes}")
        
        print(f"\n=== Memory State ===")
        print(f"Banned tools: {session.memory_state.banned_tools}")
        print(f"Successful tools: {session.memory_state.successful_tools}")
        
        print(f"\n=== Tool Performance ===")
        for record in session.tool_performance.records[-5:]:  # Last 5
            status = "[OK]" if record.success else "[X]"
            print(
                f"  {status} {record.tool_name}: {record.latency_ms}ms "
                f"(attempt for step {record.step_id})"
            )
        
        return session


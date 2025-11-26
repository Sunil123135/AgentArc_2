Orchestrated Agents Architecture - Network Diagram


================================================================================
                              USER QUERY
================================================================================
                                      +
                                      |
                                      v
================================================================================
                            COORDINATOR
================================================================================
                                      +
                                      |
        +-----------------------------+-----------------------------+
        |                             |                             |
        v                             v                             v
+-------------------+    +-------------------+    +-------------------+
|  PERCEPTION       |    |  MEMORY AGENT     |    |  RETRIEVER       |
|  AGENT            |    |                   |    |  AGENT           |
|                   |    |                   |    |                   |
|  analyze_query()  |    |  attach_relevant_ |    |  retrieve()      |
|  analyze_step_    |    |  memory()         |    |                   |
|  result()         |    |  update_from_     |    |                   |
|                   |    |  step()           |    |                   |
+-------------------+    +-------------------+    +-------------------+
        +                             +                             +
        |                             |                             |
        v                             v                             v
================================================================================
                    SESSION STATE (BLACKBOARD)
                    - PerceptionSnapshots
                    - RetrievalBundles
                    - MemoryState
                    - ToolPerformanceLog
================================================================================
                                      +
                                      |
                                      v
+-------------------+    +-------------------+    +-------------------+
|  DECISION         |    |  CRITIC AGENT     |    |  EXECUTOR        |
|  AGENT            |    |                   |    |                   |
|                   |    |                   |    |                   |
|  plan_initial()   |    |  review_result()  |    |  execute_step()  |
|  decide_next_     |    |                   |    |                   |
|  step()           |    |                   |    |                   |
|  rewrite_plan()   |    |                   |    |                   |
+-------------------+    +-------------------+    +-------------------+
        +                             +                             +
        |                             |                             |
        v                             v                             v
================================================================================
                            EXECUTION LOOP
================================================================================
                                      +
                                      |
                                      v
+-------------------+    +-------------------+    +-------------------+
|  PLAN STEP        |    |  TOOL EXECUTION   |    |  STEP RESULT     |
|  EXECUTION        |    |  (Safe Executor)   |    |  ANALYSIS        |
|                   |    |                   |    |                   |
|  - RETRIEVE       |    |  - Validation     |    |  - Quality Check |
|  - EXECUTE        |    |  - Safety Checks  |    |  - Safety Review |
|  - SUMMARIZE      |    |  - Timeout        |    |  - Accept/Reject |
|  - ASK_USER       |    |  - Retry Logic    |    |                   |
+-------------------+    +-------------------+    +-------------------+
                                      +
                                      |
        +-----------------------------+-----------------------------+
        |                             |                             |
        v                             v                             v
+-------------------+    +-------------------+    +-------------------+
|  SUCCESS PATH     |    |  FAILURE PATH     |    |  HUMAN-IN-LOOP   |
|                   |    |                   |    |                   |
|  - Advance Plan   |    |  - Mark Failed    |    |  - Tool Failure  |
|  - Update Memory  |    |  - Rewrite Plan   |    |  - Step Failure  |
|  - Continue Loop   |    |  - Retry Step     |    |  - Plan Failure  |
+-------------------+    +-------------------+    +-------------------+
                                      +
                                      |
                                      v
================================================================================
                            FINAL ANSWER
================================================================================



Detailed Component Flow Diagram

================================================================================
                        INITIALIZATION PHASE
================================================================================

USER QUERY
    +
    |
    v
COORDINATOR.run()
    +
    |
    +-----> PERCEPTION AGENT.analyze_query()
    |           +
    |           |
    |           v
    |       PerceptionSnapshot
    |           +
    |           |
    +-----------+
                |
                v
    +-----> MEMORY AGENT.attach_relevant_memory()
    |           +
    |           |
    |           v
    |       MemoryState
    |           +
    |           |
    +-----------+
                |
                v
    +-----> RETRIEVER AGENT.retrieve() [if needed]
    |           +
    |           |
    |           v
    |       RetrievalBundle
    |           +
    |           |
    +-----------+
                |
                v
    +-----> DECISION AGENT.plan_initial()
                +
                |
                v
            PlanVersion
                +
                |
                v
================================================================================
                        EXECUTION LOOP PHASE
================================================================================

LOOP START
    +
    |
    v
DECISION AGENT.decide_next_step()
    +
    |
    v
PlanStep
    +
    |
    +-----> [IF ASK_USER] -> user_input_callback()
    |           +
    |           |
    +-----------+
    |
    +-----> [IF SUMMARIZE] -> Collect results -> Final Answer
    |           +
    |           |
    +-----------+
    |
    v
EXECUTOR.execute_step()
    +
    |
    +-----> Tool Registry Validation
    +-----> Input Schema Validation
    +-----> Safety Pattern Checks
    +-----> Execute Tool (with timeout)
    +-----> Output Schema Validation
    |
    +-----> [SUCCESS] -> result_text, payload, ToolPerfRecord
    |
    +-----> [FAILURE] -> Exception
            +
            |
            +-----> ToolFailureError -> _handle_tool_failure() -> HUMAN-IN-LOOP
            |
            +-----> StepFailureError -> _handle_step_failure() -> HUMAN-IN-LOOP
    |
    v
PERCEPTION AGENT.analyze_step_result()
    +
    |
    v
PerceptionSnapshot (step_result)
    +
    |
    v
CRITIC AGENT.review_result()
    +
    |
    +-----> Quality Score Calculation
    +-----> Hallucination Risk Detection
    +-----> Safety Checks
    +-----> Acceptability Decision
    |
    v
CriticReport
    +
    |
    +-----> [IF requires_human_input] -> human_input_callback()
    |           +
    |           |
    +-----------+
    |
    +-----> [IF is_acceptable] -> Mark SUCCESS -> Advance Plan
    |
    +-----> [IF NOT acceptable] -> Mark FAILED
            +
            |
            +-----> [IF rewrite allowed] -> DECISION AGENT.rewrite_plan()
            |           +
            |           |
            |           v
            |       New PlanVersion
            |           +
            |           |
            +-----------+
            |
            +-----> [IF max rewrites reached] -> _trigger_human_in_loop()
                        +
                        |
                        v
                    PLAN FAILURE -> HUMAN-IN-LOOP
    |
    v
MEMORY AGENT.update_from_step()
    +
    |
    +-----> Record tool success/failure
    +-----> Update banned_tools (if failures >= 3)
    +-----> Store patterns from critic
    +-----> Update tool performance notes
    |
    v
Updated MemoryState
    +
    |
    v
CHECK LOOP CONDITIONS
    +
    |
    +-----> [IF session.done] -> EXIT LOOP
    |
    +-----> [IF step_count >= max_steps] -> EXIT LOOP
    |
    +-----> [IF plan.status == COMPLETED] -> EXIT LOOP
    |
    +-----> [ELSE] -> CONTINUE LOOP (back to decide_next_step)
    |
    v
================================================================================
                        FINALIZATION PHASE
================================================================================

FINAL ANSWER
    +
    |
    v
_persist_tool_performance_log()
    +
    |
    v
Save to tool_logs/*.json


Human-in-Loop Escalation Paths

================================================================================
                    HUMAN-IN-LOOP ESCALATION PATHS
================================================================================

TOOL FAILURE PATH
    +
    |
    v
ToolFailureError (hil_category="tool_failure")
    +
    |
    v
_handle_tool_failure()
    +
    |
    v
human_input_callback("Tool {name} failed...")
    +
    |
    v
Record HIL event -> Continue execution

STEP FAILURE PATH
    +
    |
    v
Step marked FAILED
    +
    |
    v
_handle_step_failure()
    +
    |
    v
human_input_callback("Step '{desc}' failed...")
    +
    |
    v
Record HIL event -> Trigger plan rewrite

PLAN FAILURE PATH
    +
    |
    v
Max rewrites/steps reached
    +
    |
    v
_trigger_human_in_loop()
    +
    |
    v
human_input_callback("Plan escalation triggered...")
    +
    |
    v
Record escalation -> Mark session.done = True



Component Interaction Summary
================================================================================
                    COMPONENT INTERACTIONS
================================================================================

COORDINATOR
    +
    |
    +-----> Calls Perception Agent
    +-----> Calls Memory Agent
    +-----> Calls Retriever Agent
    +-----> Calls Decision Agent
    +-----> Calls Executor
    +-----> Calls Critic Agent
    |
    +-----> Manages SessionState (blackboard)
    +-----> Controls execution loop
    +-----> Handles human-in-loop escalations

SESSION STATE (BLACKBOARD)
    +
    |
    +-----> Stores PerceptionSnapshots
    +-----> Stores RetrievalBundles
    +-----> Stores PlanVersions
    +-----> Stores MemoryState
    +-----> Stores ToolPerformanceLog
    |
    +-----> Shared by all agents
    +-----> Enables agent collaboration

AGENTS (Read/Write to SessionState)
    +
    |
    +-----> Perception Agent -> Writes PerceptionSnapshots
    +-----> Memory Agent -> Reads/Writes MemoryState
    +-----> Retriever Agent -> Writes RetrievalBundles
    +-----> Decision Agent -> Reads Perception/Retrieval/Memory
    |                        -> Writes PlanVersions
    +-----> Critic Agent -> Reads Step/Perception/Retrieval
    |                    -> Writes CriticReports
    +-----> Executor -> Reads PlanSteps
    |                -> Writes ToolPerfRecords


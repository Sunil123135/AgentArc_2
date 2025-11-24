# Orchestrated Agents Architecture

## Overview

The Orchestrated Agents framework implements a **Coordinator-Centric Multi-Agent Loop** pattern, inspired by AutoGen and similar multi-agent frameworks. The system uses a blackboard pattern where specialized agents collaborate through a shared `SessionState` to solve complex tasks.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      COORDINATOR                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Main Orchestration Loop                                 │  │
│  │  - Strategy Profile Management                           │  │
│  │  - Step Execution Control                                │  │
│  │  - Retry & Timeout Logic                                 │  │
│  │  - Parallel Strategy Execution                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────┬─────────────────────────────────────────────────┘
               │
               ├─────────────────────────────────────────────┐
               │                                               │
               ▼                                               ▼
┌──────────────────────────┐                    ┌──────────────────────────┐
│   PERCEPTION AGENT       │                    │   RETRIEVER AGENT         │
│                          │                    │                          │
│ Input Heuristics:        │                    │ Input Heuristics:         │
│ - Query length (3-5000)   │                    │ - Query length (3-500)    │
│ - Suspicious patterns     │                    │ - Pattern validation      │
│ - Entity extraction       │                    │                          │
│                          │                    │ Output Heuristics:        │
│ Output Heuristics:       │                    │ - Max items (20)          │
│ - Max entities (50)      │                    │ - Relevance scores        │
│ - Max sub-goals (10)      │                    │ - Summary validation      │
│ - Confidence (0-1)       │                    │                          │
│                          │                    │ Strategies:               │
│ Output:                  │                    │ - Memory retrieval      │
│ - PerceptionSnapshot     │                    │ - Vector store           │
│   - Entities             │                    │ - Web search             │
│   - Intent               │                    │ - File search            │
│   - Constraints          │                    │                          │
│   - Sub-goals            │                    │ Output:                   │
│   - Confidence           │                    │ - RetrievalBundle        │
└──────────┬───────────────┘                    └──────────┬───────────────┘
           │                                               │
           │                                               │
           ▼                                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SESSION STATE (Blackboard)                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  - PerceptionSnapshots                                   │  │
│  │  - RetrievalBundles                                      │  │
│  │  - PlanVersions                                          │  │
│  │  - MemoryState                                           │  │
│  │  - ToolPerformanceLog                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────┬─────────────────────────────────────────────────┘
               │
               ├─────────────────────────────────────────────┐
               │                                               │
               ▼                                               ▼
┌──────────────────────────┐                    ┌──────────────────────────┐
│   MEMORY AGENT           │                    │   DECISION AGENT          │
│                          │                    │                          │
│ Input Heuristics:        │                    │ Input Heuristics:         │
│ - Perception validation  │                    │ - Perception validation    │
│ - Step results           │                    │ - Retrieval validation    │
│                          │                    │ - Memory state            │
│ Output Heuristics:       │                    │                          │
│ - Max short-term (100)   │                    │ Output Heuristics:        │
│ - Max banned tools (50)  │                    │ - Max steps (20)          │
│ - Max successful (100)   │                    │ - Step description length │
│                          │                    │ - Plan rewrite limits      │
│ Output:                  │                    │                          │
│ - MemoryState            │                    │ Output:                   │
│   - Short-term memory    │                    │ - PlanVersion             │
│   - Banned tools         │                    │   - Steps                 │
│   - Successful tools     │                    │   - Current index         │
│   - Long-term refs       │                    │   - Status                │
└──────────┬───────────────┘                    └──────────┬───────────────┘
           │                                               │
           │                                               │
           ▼                                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PLAN EXECUTION                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  PlanStep                                                │  │
│  │  - Tool name                                             │  │
│  │  - Tool arguments                                         │  │
│  │  - Expected outcome                                       │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────┬─────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SAFE EXECUTOR                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Safety Checks:                                           │  │
│  │  ✅ Tool registry validation                              │  │
│  │  ✅ Input schema validation (Pydantic)                    │  │
│  │  ✅ Dangerous pattern blocking                            │  │
│  │     - import, open, exec, eval, compile                  │  │
│  │     - Path traversal, dangerous commands                  │  │
│  │  ✅ Timeout (3-5 seconds)                                 │  │
│  │  ✅ Retry logic (max 3 attempts)                          │  │
│  │  ✅ Output schema validation                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────┬─────────────────────────────────────────────────┘
               │
               ├─────────────────────────────────────────────┐
               │                                               │
               ▼                                               ▼
┌──────────────────────────┐                    ┌──────────────────────────┐
│   TOOL REGISTRY          │                    │   PARALLEL STRATEGY      │
│                          │                    │   EXECUTOR               │
│ - Tool registration      │                    │                          │
│ - Schema validation      │                    │ Strategies:              │
│ - Safety checks          │                    │ - Tool execution          │
│                          │                    │ - RAG query              │
│ Registered Tools:        │                    │ - Web search             │
│ - python (validated)     │                    │                          │
│ - calculator            │                    │ Execution Modes:         │
│ - web_search            │                    │ - EXPLORATORY: Parallel │
│ - file_read             │                    │ - FALLBACK: Sequential   │
│                          │                    │ - CONSERVATIVE: Tool only│
│ Blocked Operations:      │                    │                          │
│ ❌ import, open, exec    │                    │ Output:                  │
│ ❌ eval, compile         │                    │ - Best result selection   │
│ ❌ Path traversal        │                    │ - Comparison metrics      │
└──────────┬───────────────┘                    └──────────┬───────────────┘
           │                                               │
           │                                               │
           ▼                                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CRITIC AGENT                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Input Heuristics:                                        │  │
│  │  - Step result validation                                 │  │
│  │  - Perception validation                                  │  │
│  │  - Retrieval validation                                   │  │
│  │                                                           │  │
│  │  Output Heuristics:                                       │  │
│  │  - Quality score (0-1)                                    │  │
│  │  - Hallucination risk (0-1)                              │  │
│  │  - Max issues (20)                                       │  │
│  │                                                           │  │
│  │  Output:                                                  │  │
│  │  - CriticReport                                           │  │
│  │    - is_acceptable                                        │  │
│  │    - issues                                               │  │
│  │    - rewrite_suggestion                                  │  │
│  │    - requires_human_input                                │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────┬─────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL ANSWER                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Coordinator

The **Coordinator** is the central orchestrator that manages the entire agent loop. It:

- **Manages Strategy Profiles**: Controls behavior based on CONSERVATIVE, EXPLORATORY, or FALLBACK strategies
- **Coordinates Agent Execution**: Calls agents in the correct sequence
- **Handles Retries**: Implements retry logic with exponential backoff
- **Manages Timeouts**: Enforces timeout limits per step
- **Parallel Execution**: Coordinates parallel strategy execution for exploratory mode
- **Plan Management**: Tracks plan versions and rewrites

### 2. Perception Agent

**Purpose**: Analyzes user queries and step results to extract structured information.

**Input Heuristics**:
- Query length: 3-5000 characters
- Suspicious pattern detection (XSS, code injection)
- Entity extraction (names, numbers, quoted strings)

**Output Heuristics**:
- Maximum 50 entities per snapshot
- Maximum 10 sub-goals
- Confidence score between 0.0 and 1.0

**Output**: `PerceptionSnapshot` with entities, intent, constraints, sub-goals, and confidence.

### 3. Retriever Agent

**Purpose**: Retrieves relevant information from multiple sources.

**Input Heuristics**:
- Query length: 3-500 characters
- Pattern validation

**Output Heuristics**:
- Maximum 20 retrieved items
- Relevance scores between 0.0 and 1.0
- Summary validation

**Strategies**:
- Memory retrieval (short-term session memory)
- Vector store search
- Web search (optional)
- File system search

**Output**: `RetrievalBundle` with items, relevance scores, and summaries.

### 4. Memory Agent

**Purpose**: Manages short-term and long-term memory.

**Input Heuristics**:
- Perception validation
- Step result validation

**Output Heuristics**:
- Maximum 100 short-term memory items
- Maximum 50 banned tools
- Maximum 100 successful tool entries

**Output**: `MemoryState` with short-term memory, banned tools, and successful tool patterns.

### 5. Decision Agent

**Purpose**: Creates and manages execution plans.

**Input Heuristics**:
- Perception validation
- Retrieval validation
- Memory state validation

**Output Heuristics**:
- Maximum 20 steps per plan
- Minimum 5 characters per step description
- Maximum 5 plan rewrites

**Output**: `PlanVersion` with steps, current index, and status.

### 6. Safe Executor

**Purpose**: Executes plan steps with safety checks, validation, and retry logic.

**Safety Features**:
- ✅ **Tool Registry Validation**: Only registered tools can be executed
- ✅ **Input Schema Validation**: Pydantic-based input validation
- ✅ **Dangerous Pattern Blocking**: Blocks `import`, `open`, `exec`, `eval`, `compile`, path traversal
- ✅ **Timeout**: 3-5 seconds per tool call (configurable)
- ✅ **Retry Logic**: Maximum 3 retry attempts with exponential backoff
- ✅ **Output Schema Validation**: Validates output against expected schema
- ✅ **Max Tool Calls**: Limits tool calls per script (configurable)

**Blocked Operations**:
- ❌ `import`, `from ... import`
- ❌ `open()`, `file()`
- ❌ `exec()`, `eval()`, `compile()`
- ❌ `__import__()`
- ❌ `input()`, `raw_input()`
- ❌ Path traversal (`../`)
- ❌ Dangerous commands (`rm -rf`, `format`, etc.)

### 7. Tool Registry

**Purpose**: Central registry for tool management and validation.

**Features**:
- Tool registration with schema definition
- Input/output validation using Pydantic
- Safety pattern checking
- Timeout and retry configuration per tool
- Tool discovery and listing

**Example Tool Registration**:
```python
registry = ToolRegistry()
registry.register_simple(
    name="calculator",
    description="Perform mathematical calculations",
    executor=calculator_function,
    input_schema={"expression": {"type": "string"}},
    timeout_seconds=3.0,
    max_retries=3,
)
```

### 8. Parallel Strategy Executor

**Purpose**: Execute multiple strategies in parallel and compare results.

**Strategies**:
1. **Tool Execution**: Direct tool call
2. **RAG Query**: Retrieval-Augmented Generation from knowledge base
3. **Web Search**: External web search (optional)

**Execution Modes**:

- **EXPLORATORY**: All strategies run in parallel, best result selected
- **FALLBACK**: Strategies run sequentially until one succeeds
- **CONSERVATIVE**: Only tool execution (no parallel strategies)

**Result Comparison**:
- Priority: Tool > RAG > Web Search
- Latency tracking
- Success rate tracking
- Best result selection

### 9. Critic Agent

**Purpose**: Reviews step results for quality, safety, and correctness.

**Input Heuristics**:
- Step result validation
- Perception validation
- Retrieval validation

**Output Heuristics**:
- Quality score: 0.0-1.0
- Hallucination risk: 0.0-1.0
- Maximum 20 issues per report

**Evaluation Criteria**:
- Result matches expected outcome
- Retrieval overlap
- Hallucination detection
- Safety flag detection

**Output**: `CriticReport` with quality score, issues, and rewrite suggestions.

## Strategy Profiles

### CONSERVATIVE
- **Max Steps**: 5
- **Max Retries**: 2
- **Max Plan Rewrites**: 1
- **Parallel Strategies**: Disabled
- **Use Case**: Fast, focused execution with minimal tool usage

### EXPLORATORY
- **Max Steps**: 15
- **Max Retries**: 3
- **Max Plan Rewrites**: 3
- **Parallel Strategies**: Enabled (all strategies run in parallel)
- **Use Case**: Comprehensive exploration with multiple approaches

### FALLBACK
- **Max Steps**: 10
- **Max Retries**: 4
- **Max Plan Rewrites**: 2
- **Parallel Strategies**: Sequential fallback (try tool, then RAG, then web search)
- **Use Case**: Robust execution with multiple fallback options

## Data Flow

1. **User Query** → Coordinator
2. **Coordinator** → Perception Agent → `PerceptionSnapshot`
3. **Coordinator** → Memory Agent → `MemoryState`
4. **Coordinator** → Retriever Agent → `RetrievalBundle` (if needed)
5. **Coordinator** → Decision Agent → `PlanVersion`
6. **Coordinator** → Safe Executor → Execute `PlanStep`
   - Tool Registry validation
   - Input validation
   - Safety checks
   - Timeout & retry
   - Output validation
7. **Coordinator** → Perception Agent → Analyze result
8. **Coordinator** → Critic Agent → `CriticReport`
9. **Coordinator** → Memory Agent → Update memory
10. **If rewrite needed**: Decision Agent → New `PlanVersion`
11. **If done**: Coordinator → Final Answer

## Safety Architecture

### Multi-Layer Safety

1. **Tool Registry Layer**: Only registered tools can be executed
2. **Input Validation Layer**: Pydantic schema validation
3. **Pattern Matching Layer**: Dangerous pattern detection
4. **Timeout Layer**: Prevents infinite execution
5. **Retry Layer**: Handles transient failures
6. **Output Validation Layer**: Ensures output matches expected schema

### Validation Points

- **Perception Input**: Query length, suspicious patterns
- **Retriever Input**: Query validation, pattern checking
- **Tool Input**: Schema validation, safety checks
- **Agent Output**: Heuristic validation (max items, scores, etc.)
- **Tool Output**: Schema validation

## Integration with AutoGen

This framework follows similar principles to AutoGen:

- **Multi-Agent Collaboration**: Multiple specialized agents work together
- **Conversation Patterns**: Agents communicate through structured messages
- **Tool Use**: Safe tool execution with validation
- **Human-in-the-Loop**: Support for human feedback and approval

However, this framework adds:
- **Stronger Safety**: Built-in safety checks and validation
- **Pydantic Validation**: Type-safe data structures throughout
- **Parallel Strategies**: Multiple execution strategies in parallel
- **Tool Registry**: Centralized tool management

## JSON Prompt Contracts

Independent prompts keep each component’s responsibilities explicit whether the implementation is heuristic or LLM-backed. The canonical prompts live in `src/orchestrated_agents/prompts/*.json` and are loaded via `agent_prompts.py`. Override them by editing the JSON files or by passing a custom string/`AgentPrompt` when constructing a component.

| Component  | Prompt Focus                                                                                           |
|------------|---------------------------------------------------------------------------------------------------------|
| Perception | Interpret raw text, extract entities/intent/constraints, detect goal satisfaction, avoid tool use.      |
| Retriever  | Surface verifiable context snippets plus open questions that highlight knowledge gaps.                  |
| Memory     | Maintain short-term history, tool performance summaries, and critic-derived behavioral patterns.        |
| Decision   | Produce bounded multi-step tool plans with explicit tool choices, finishing with a summarize step.      |
| Critic     | Score quality and safety, log concrete issues, and escalate to humans for high-risk or unclear output.  |
| Executor   | Execute only registered tools with validated inputs, respect timeouts/retries, and emit diagnostics.    |

Supplying an `llm_completion` callable causes the framework to prepend the JSON prompt instructions to the runtime input and append the resulting guidance into the component’s notes (e.g., perception notes, retriever summaries, memory annotations, critic issues, or safe-executor error payloads).

## Extensibility

### Adding New Agents

1. Inherit from the appropriate base class in `interfaces.py`
2. Implement required methods
3. Add heuristics to `AgentHeuristics`
4. Register with Coordinator

### Adding New Tools

1. Define tool schema with `ToolSchema`
2. Register with `ToolRegistry`
3. Implement executor function
4. Configure safety rules

### Custom Strategies

1. Extend `StrategyProfile` enum
2. Add limits to `default_limits()` in `strategy.py`
3. Implement parallel strategy logic if needed

## Performance Considerations

- **Parallel Execution**: ThreadPoolExecutor for concurrent strategy execution
- **Timeout Management**: Prevents hanging operations
- **Retry Logic**: Exponential backoff to avoid thundering herd
- **Memory Limits**: Heuristics prevent unbounded memory growth
- **Tool Call Limits**: Configurable limits per script

## Security Considerations

- **Input Sanitization**: All inputs validated before processing
- **Pattern Blocking**: Dangerous operations blocked at multiple layers
- **Timeout Enforcement**: Prevents resource exhaustion
- **Tool Isolation**: Only registered tools can execute
- **Output Validation**: Ensures outputs match expected schemas


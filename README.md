# Orchestrated Agents

A Coordinator-Centric Multi-Agent Loop Framework for Python 3.11+ with built-in safety, validation, and parallel strategy execution.

## Features

✅ **Built-in Safety**: Tool registry, input/output validation, dangerous operation blocking  
✅ **Pydantic Validation**: Type-safe data structures throughout the framework  
✅ **Timeout & Retry**: Configurable timeouts (3-5s) and retry logic (max 3 attempts)  
✅ **Parallel Strategies**: Execute multiple strategies (tool, RAG, web search) in parallel  
✅ **Agent Heuristics**: Input/output validation for each agent  
✅ **Strategy Profiles**: CONSERVATIVE, EXPLORATORY, and FALLBACK modes  
✅ **Tool Registry**: Centralized tool management with schema validation  
✅ **AutoGen-Inspired**: Multi-agent collaboration patterns similar to AutoGen  

## Architecture

This framework implements a multi-agent system where a central **Coordinator** orchestrates specialized agents that work together to solve complex tasks. The architecture follows a blackboard pattern where all agents share a common `SessionState` (the blackboard) containing typed snapshots of their work.

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

### Core Components

```
Coordinator
  ↙  ↓  ↓  ↘
Perception  Retriever  Critic  Memory
  ↓    ↓      ↑    ↓
Plan/Step → Executor ← Plan Rewrite
  ↑
Decision Agent
```

### Agent Responsibilities

1. **Coordinator**: Owns the main loop, maintains global state, enforces limits, and coordinates all other agents.

2. **Perception Agent**: Analyzes user queries and step results to extract:
   - Entities (names, numbers, quoted strings)
   - Intent (question, command, analysis, creation)
   - Constraints (time, safety, tool restrictions)
   - Sub-goals and uncertainties
   - Goal satisfaction status

3. **Retriever Agent**: Retrieves relevant information from:
   - Memory (short-term session memory)
   - Knowledge base (files, vector stores)
   - Web search (optional)
   - Returns structured `RetrievalBundle` with items and summaries

4. **Memory Agent**: Manages short and long-term memory:
   - Attaches relevant past context to current perception
   - Records tool successes and failures
   - Bans tools after repeated failures
   - Tracks successful tool usage patterns

5. **Decision Agent**: Creates and manages execution plans:
   - `plan_initial`: Creates initial multi-step plan from perception
   - `decide_next_step`: Selects next step from active plan
   - `rewrite_plan`: Rewrites plan after failures with alternative approaches

6. **Critic Agent**: Reviews step results:
   - Evaluates quality against expected outcomes
   - Detects hallucination risks
   - Flags safety concerns
   - Decides if step is acceptable or needs rewrite
   - Can request human input for critical decisions

7. **Executor**: Executes plan steps (wraps existing MCP/executor code):
   - Takes `PlanStep` with tool name and arguments
   - Returns result text, payload, and performance metrics
   - Handles errors and logs performance

### Agent & Executor Prompts

Each component ships with its own JSON prompt stored under `src/orchestrated_agents/prompts/NAME.json`. These prompts describe the role-specific guardrails used when the component calls an LLM (via the optional `llm_completion` hook) or when you need to audit the behavioral contract. Highlights:

- **Perception** – “Sensemaker” that extracts structured entities, intent, constraints, and satisfaction signals without inventing execution steps.
- **Retriever** – “Context gatherer” that surfaces trusted snippets plus open questions, avoiding hallucinated facts.
- **Memory** – “Historian” that records tool performance, critic insights, and augments session notes with aggregated statistics.
- **Decision** – “Planner” that authors bounded multi-step tool plans, always ending with a summarize action.
- **Critic** – “Quality reviewer” that scores each step, documents issues, and escalates high-risk outputs to humans.
- **Executor** – “Guardrailed tool runner” that reminds the safety layer to execute only registered tools with validated inputs.

To override a prompt you can either edit the JSON file (for project-wide changes) or pass a custom string/`AgentPrompt` when instantiating the component:

```python
from orchestrated_agents.decision_agent import SimpleDecisionAgent

decision = SimpleDecisionAgent(prompt="You are an automation strategist. Prefer calculators for pure math.")
```

When you supply an `llm_completion` callable, the framework automatically injects the corresponding prompt instructions before invoking the LLM and appends the guidance back into notes/diagnostics (e.g., memory notes, retriever open questions, safe-executor error payloads). Use `list_default_prompts()` to inspect the current JSON-backed defaults at runtime.

### Data Flow

All agents communicate through typed data structures in `SessionState`:

- **PerceptionSnapshot**: Structured analysis of queries/results
- **RetrievalBundle**: Retrieved information with relevance scores
- **PlanVersion**: Versioned execution plans with steps
- **PlanStep**: Individual steps with status, results, and metadata
- **CriticReport**: Quality assessment and recommendations
- **MemoryState**: Short-term memory items and tool performance tracking
- **ToolPerformanceLog**: Performance metrics for all tool executions

## Sequence Diagram

```
User Query
    ↓
Coordinator.run()
    ↓
PerceptionAgent.analyze_query() → PerceptionSnapshot
    ↓
MemoryAgent.attach_relevant_memory() → MemoryState
    ↓
RetrieverAgent.retrieve() → RetrievalBundle (if needed)
    ↓
DecisionAgent.plan_initial() → PlanVersion
    ↓
┌─────────────────────────────────────┐
│ Main Loop (for each step):          │
│                                     │
│  DecisionAgent.decide_next_step()  │
│         ↓                           │
│  Executor.execute_step()            │
│         ↓                           │
│  PerceptionAgent.analyze_step_result()│
│         ↓                           │
│  CriticAgent.review_result()        │
│         ↓                           │
│  MemoryAgent.update_from_step()     │
│         ↓                           │
│  If needs_rewrite:                  │
│    DecisionAgent.rewrite_plan()     │
│         ↓                           │
│  If done:                           │
│    SessionState.mark_done()         │
└─────────────────────────────────────┘
    ↓
Final Answer
```

## Installation

```bash
# Install dependencies
pip install -e .

# Or using uv (recommended)
uv pip install -e .
```

## Usage

### Basic Usage

```python
python main.py "Calculate the sum of 1 to 100"
```

The CLI uses the default coordinator stack wired up in `main.py`, but you can still import the individual components if you want to embed them elsewhere.

```python
from orchestrated_agents import Coordinator, StrategyProfile
from orchestrated_agents.decision_agent import SimpleDecisionAgent
...
```

### Using Parallel Strategies

```python
from orchestrated_agents import ParallelStrategyExecutor

# Define RAG executor
def rag_query(query: str, session: SessionState) -> RetrievalBundle:
    # Your RAG implementation
    pass

# Define web search executor
def web_search(query: str, session: SessionState) -> dict:
    # Your web search implementation
    pass

# Create parallel strategy executor
parallel_executor = ParallelStrategyExecutor(
    tool_executor=executor.execute_step,
    rag_executor=rag_query,
    web_search_executor=web_search,
)

# Execute with parallel strategies (for EXPLORATORY mode)
results = parallel_executor.execute_parallel(step, session)
comparison = parallel_executor.compare_results(results)
print(f"Best strategy: {comparison['selected_strategy']}")
```

### Using Agent Heuristics

```python
from orchestrated_agents import AgentHeuristics

# Create heuristics
heuristics = AgentHeuristics()

# Validate perception input
is_valid, error = heuristics.perception.validate_query("Calculate 2+2")
if not is_valid:
    print(f"Invalid query: {error}")

# Validate perception output
is_valid, error = heuristics.perception.validate_perception_output(perception_snapshot)
if not is_valid:
    print(f"Invalid perception: {error}")
```

### Demo CLI

```bash
# Run with default (conservative) strategy
python -m orchestrated_agents.demo_cli "Your query here"

# Run with exploratory strategy
python -m orchestrated_agents.demo_cli exploratory "Your query here"

# Run with fallback strategy
python -m orchestrated_agents.demo_cli fallback "Your query here"
```

Or interactively:

```bash
python -m orchestrated_agents.demo_cli
# Enter your query when prompted
```

## Safety Features

### Built-in Safety Checks

✅ **Tool Registry**: Only registered tools can be executed  
✅ **Input Validation**: Pydantic schema validation for all inputs  
✅ **Output Validation**: Schema validation for tool outputs  
✅ **Dangerous Pattern Blocking**: Blocks `import`, `open`, `exec`, `eval`, `compile`  
✅ **Timeout Enforcement**: 3-5 seconds per tool call (configurable)  
✅ **Retry Logic**: Maximum 3 retry attempts with exponential backoff  
✅ **Max Tool Calls**: Configurable limit on tool calls per script  

### Blocked Operations

The framework automatically blocks dangerous operations:
- ❌ `import`, `from ... import`
- ❌ `open()`, `file()`
- ❌ `exec()`, `eval()`, `compile()`
- ❌ `__import__()`
- ❌ `input()`, `raw_input()`
- ❌ Path traversal (`../`)
- ❌ Dangerous commands (`rm -rf`, `format`, etc.)

## Strategy Profiles

The framework supports three strategy profiles:

1. **CONSERVATIVE**: 
   - Max 5 steps, 2 retries, 1 plan rewrite
   - Minimal tool usage
   - Fast, focused execution
   - **Parallel Strategies**: Disabled (tool execution only)

2. **EXPLORATORY**:
   - Max 15 steps, 3 retries, 3 plan rewrites
   - More retrieval and parallel exploration
   - More critic involvement
   - **Parallel Strategies**: Enabled (tool, RAG, web search run in parallel)

3. **FALLBACK**:
   - Max 10 steps, 4 retries, 2 plan rewrites
   - Prioritizes robust tools
   - More memory usage and human-in-the-loop
   - **Parallel Strategies**: Sequential fallback (try tool, then RAG, then web search)

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=orchestrated_agents
```

## Project Structure

```
orchestrated-agents/
├── pyproject.toml              # Project configuration
├── README.md                   # This file
├── ARCHITECTURE.md             # Detailed architecture documentation
├── src/
│   └── orchestrated_agents/
│       ├── __init__.py
│       ├── models.py              # Shared data models (blackboard)
│       ├── strategy.py            # Strategy profiles
│       ├── interfaces.py         # Abstract base classes
│       ├── coordinator.py         # Main orchestration loop
│       ├── perception_agent.py    # Perception agent implementation
│       ├── retriever_agent.py     # Retriever agent implementation
│       ├── memory_agent.py        # Memory agent implementation
│       ├── decision_agent.py      # Decision agent implementation
│       ├── critic_agent.py        # Critic agent implementation
│       ├── executor_adapter.py    # Base executor adapters
│       ├── tool_registry.py       # Tool registry with validation
│       ├── safe_executor.py       # Safe executor with timeout/retry
│       ├── agent_heuristics.py    # Input/output heuristics
│       ├── parallel_strategy.py   # Parallel strategy execution
│       └── demo_cli.py            # CLI demo
└── tests/
    ├── test_models.py
    ├── test_coordinator_happy_path.py
    ├── test_coordinator_failure_replan.py
    └── test_memory_and_tool_performance.py
```

## Extending the Framework

### Creating Custom Agents

All agents implement abstract interfaces defined in `interfaces.py`. To create a custom agent:

1. Inherit from the appropriate base class (e.g., `PerceptionAgent`)
2. Implement all abstract methods
3. Pass to `Coordinator` constructor

Example:

```python
from orchestrated_agents.interfaces import PerceptionAgent
from orchestrated_agents.models import PerceptionSnapshot, PlanStep, SessionState

class MyCustomPerceptionAgent(PerceptionAgent):
    def analyze_query(self, user_query: str, session: SessionState) -> PerceptionSnapshot:
        # Your implementation
        pass
    
    def analyze_step_result(self, step: PlanStep, raw_output: str, session: SessionState) -> PerceptionSnapshot:
        # Your implementation
        pass
```

### Integrating with LLMs

The framework is designed to be LLM-agnostic. Agents accept optional `llm_completion` callables in their constructors. To integrate with an LLM:

```python
def my_llm_completion(prompt: str) -> str:
    # Call your LLM API
    return llm_response

perception_agent = RuleBasedPerceptionAgent(llm_completion=my_llm_completion)
```

### Custom Executors

To integrate with an existing executor:

```python
from orchestrated_agents.executor_adapter import MCPExecutor

# Your existing executor instance
my_executor = MyExistingExecutor()

# Wrap it
executor = MCPExecutor(my_executor)

# Use in coordinator
coordinator = Coordinator(..., executor=executor, ...)
```

## Key Design Principles

1. **Typed Snapshots**: All data passed between agents is strongly typed using Pydantic models, not loose strings.

2. **Blackboard Pattern**: All agents read/write to shared `SessionState`, enabling loose coupling.

3. **Strategy Profiles**: Behavior is configurable via strategy profiles without code changes.

4. **Versioned Plans**: Plans are versioned, allowing rewrites while preserving history.

5. **Observable State**: All state changes are logged and traceable for debugging.

6. **Testable**: Agents are pure functions/classes with clear interfaces, making testing straightforward.

7. **Safety First**: Multiple layers of safety checks prevent dangerous operations.

8. **Validation Everywhere**: Input/output validation at every agent boundary using Pydantic.

9. **Parallel Execution**: Multiple strategies can execute in parallel for better results.

10. **AutoGen-Inspired**: Follows multi-agent collaboration patterns similar to AutoGen framework.

## License

This project is provided as-is for educational and development purposes.


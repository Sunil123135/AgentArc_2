"""Parallel strategy execution for exploratory and fallback modes."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from .models import PlanStep, RetrievalBundle, SessionState, StrategyProfile


class ParallelStrategyExecutor:
    """Execute multiple strategies in parallel and compare results."""
    
    def __init__(
        self,
        tool_executor: Callable[[PlanStep, SessionState], tuple[str, dict, Any]],
        rag_executor: Optional[Callable[[str, SessionState], RetrievalBundle]] = None,
        web_search_executor: Optional[Callable[[str, SessionState], dict[str, Any]]] = None,
    ):
        """Initialize parallel strategy executor.
        
        Args:
            tool_executor: Function to execute tool calls
            rag_executor: Function to execute RAG queries
            web_search_executor: Function to execute web searches
        """
        self.tool_executor = tool_executor
        self.rag_executor = rag_executor
        self.web_search_executor = web_search_executor
    
    def execute_parallel(
        self,
        step: PlanStep,
        session: SessionState,
        strategies: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Execute multiple strategies in parallel.
        
        Args:
            step: The plan step to execute
            session: Current session state
            strategies: List of strategies to execute (default: all available)
        
        Returns:
            Dictionary with results from each strategy
        """
        if strategies is None:
            strategies = ["tool", "rag", "web_search"]
        
        results = {}
        
        # Determine which strategies to run based on session strategy profile
        if session.strategy_profile == StrategyProfile.EXPLORATORY:
            # Run all strategies in parallel
            with ThreadPoolExecutor(max_workers=len(strategies)) as executor:
                futures = {}
                
                if "tool" in strategies:
                    futures["tool"] = executor.submit(
                        self._execute_tool, step, session
                    )
                
                if "rag" in strategies and self.rag_executor:
                    query = step.input_context or step.description
                    futures["rag"] = executor.submit(
                        self._execute_rag, query, session
                    )
                
                if "web_search" in strategies and self.web_search_executor:
                    query = step.input_context or step.description
                    futures["web_search"] = executor.submit(
                        self._execute_web_search, query, session
                    )
                
                # Collect results
                for strategy_name, future in futures.items():
                    try:
                        results[strategy_name] = future.result(timeout=10.0)
                    except Exception as e:
                        results[strategy_name] = {
                            "success": False,
                            "error": str(e),
                            "strategy": strategy_name,
                        }
        
        elif session.strategy_profile == StrategyProfile.FALLBACK:
            # Try strategies sequentially until one succeeds
            for strategy_name in strategies:
                try:
                    if strategy_name == "tool":
                        result = self._execute_tool(step, session)
                        if result.get("success"):
                            results[strategy_name] = result
                            break
                    elif strategy_name == "rag" and self.rag_executor:
                        query = step.input_context or step.description
                        result = self._execute_rag(query, session)
                        if result.get("success"):
                            results[strategy_name] = result
                            break
                    elif strategy_name == "web_search" and self.web_search_executor:
                        query = step.input_context or step.description
                        result = self._execute_web_search(query, session)
                        if result.get("success"):
                            results[strategy_name] = result
                            break
                except Exception as e:
                    results[strategy_name] = {
                        "success": False,
                        "error": str(e),
                        "strategy": strategy_name,
                    }
                    continue
        
        else:  # CONSERVATIVE
            # Only use tool execution
            if "tool" in strategies:
                results["tool"] = self._execute_tool(step, session)
        
        return results
    
    def _execute_tool(
        self, step: PlanStep, session: SessionState
    ) -> dict[str, Any]:
        """Execute tool call strategy."""
        try:
            start_time = time.time()
            result_text, result_payload, tool_perf = self.tool_executor(step, session)
            latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                "success": tool_perf.success,
                "result_text": result_text,
                "result_payload": result_payload,
                "latency_ms": latency_ms,
                "strategy": "tool",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "strategy": "tool",
            }
    
    def _execute_rag(
        self, query: str, session: SessionState
    ) -> dict[str, Any]:
        """Execute RAG query strategy."""
        if not self.rag_executor:
            return {
                "success": False,
                "error": "RAG executor not configured",
                "strategy": "rag",
            }
        
        try:
            start_time = time.time()
            bundle = self.rag_executor(query, session)
            latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                "success": True,
                "bundle": bundle,
                "items_count": len(bundle.items),
                "latency_ms": latency_ms,
                "strategy": "rag",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "strategy": "rag",
            }
    
    def _execute_web_search(
        self, query: str, session: SessionState
    ) -> dict[str, Any]:
        """Execute web search strategy."""
        if not self.web_search_executor:
            return {
                "success": False,
                "error": "Web search executor not configured",
                "strategy": "web_search",
            }
        
        try:
            start_time = time.time()
            results = self.web_search_executor(query, session)
            latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                "success": True,
                "results": results,
                "latency_ms": latency_ms,
                "strategy": "web_search",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "strategy": "web_search",
            }
    
    def compare_results(self, results: dict[str, Any]) -> dict[str, Any]:
        """Compare results from multiple strategies and select best."""
        successful_results = {
            k: v for k, v in results.items()
            if v.get("success", False)
        }
        
        if not successful_results:
            return {
                "selected_strategy": None,
                "best_result": None,
                "all_results": results,
                "comparison": "No successful strategies",
            }
        
        # Simple comparison: prefer tool results, then RAG, then web search
        priority_order = ["tool", "rag", "web_search"]
        
        for strategy in priority_order:
            if strategy in successful_results:
                return {
                    "selected_strategy": strategy,
                    "best_result": successful_results[strategy],
                    "all_results": results,
                    "comparison": f"Selected {strategy} as best strategy",
                }
        
        # Fallback: return first successful result
        first_strategy = list(successful_results.keys())[0]
        return {
            "selected_strategy": first_strategy,
            "best_result": successful_results[first_strategy],
            "all_results": results,
            "comparison": f"Selected {first_strategy} as fallback",
        }


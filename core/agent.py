"""
agent.py
--------
Core agent loop:
- route
- plan
- execute (worker models or local tools)
- judge (choose ONE final answer)
- evaluate
- store memory

IMPORTANT DESIGN RULES (for stability + self-dev):
1) Do NOT create provider clients at import time.
   - Import time happens before .env is loaded in many setups.
   - Missing API keys should NOT crash the program.
2) Providers are OPTIONAL.
   - If a key is missing, that provider is simply not registered.
3) Models are configuration, not new providers.
   - We still expose "openai_dev" / "claude_dev" as separate entries because it
     simplifies policy routing and prevents accidental use of cheap models for dev work.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from core.router import Router, RouteDecision
from core.planner import Planner, Plan
from core.memory import MemoryStore
from core.judge import Judge
from core.general_routing import (
    build_local_assessment_prompt,
    build_local_evidence_prompt,
    classify_intent,
    decide_general_route,
    is_authoritative_query,
    is_factual_query,
    parse_local_assessment,
    summarize_evidence,
)

# Local tools
from tools.local_exec import read_file, write_file, list_dir
from tools.web_search import web_search
from tools.web_search_vertex import web_search_vertex_status

# Provider clients (safe to import; actual instantiation happens in __init__)
from providers.openai_client import OpenAIClient
from providers.openai_responses_client import OpenAIResponsesClient
from providers.claude_client import ClaudeClient
from providers.ollama_client import OllamaClient

# If you have a Gemini client, import it here.
# If you DON'T have it, keep it commented out to avoid import errors.
# from providers.gemini_client import GeminiClient


class Agent:
    """
    Main orchestration agent.

    Notes:
    - We build provider_map at runtime (inside __init__) after environment variables
      are available.
    - Missing API keys simply mean that provider won't be available.
    """

    def __init__(self, capabilities: dict, memory: MemoryStore) -> None:
        self.capabilities = capabilities
        self.memory = memory

        # Core components
        self.router = Router(capabilities)
        self.planner = Planner(capabilities)

        # Build providers safely (no crashing if a key is missing)
        self.provider_map = self._build_provider_map()

        # Judge component (selects judge provider + synthesizes final answer)
        self.judge = Judge(capabilities=capabilities, provider_map=self.provider_map)

    # ----------------------------
    # Provider wiring
    # ----------------------------

    def _provider_enabled_in_capabilities(self, name: str) -> bool:
        """
        Returns True if capabilities.json marks this provider as enabled.
        If provider is not listed, treat as disabled (safer default).
        """
        providers_cfg = self.capabilities.get("providers", {})
        return bool(providers_cfg.get(name, {}).get("enabled", False))

    def _build_provider_map(self) -> Dict[str, Any]:
        """
        Create provider clients only if:
        - provider is enabled in capabilities.json
        - required API key exists in environment (for that provider)

        Returns:
            dict mapping provider_name -> client with .generate(prompt) method
        """
        provider_map: Dict[str, Any] = {}

        # ----------------------------
        # OpenAI (cheap/default)
        # ----------------------------
        if self._provider_enabled_in_capabilities("openai") and os.getenv("OPENAI_API_KEY"):
            provider_map["openai"] = OpenAIClient(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.2,
            )

        # ----------------------------
        # OpenAI (dev-grade)
        # ----------------------------
        if self._provider_enabled_in_capabilities("openai_dev") and os.getenv("OPENAI_API_KEY"):
            provider_map["openai_dev"] = OpenAIResponsesClient(
                model=os.getenv("OPENAI_DEV_MODEL", "gpt-5.2-codex"),
                reasoning_effort=os.getenv("OPENAI_DEV_REASONING_EFFORT", "high"),
            )

        # ----------------------------
        # Claude (cheap/default)
        # ----------------------------
        if self._provider_enabled_in_capabilities("claude") and os.getenv("ANTHROPIC_API_KEY"):
            provider_map["claude"] = ClaudeClient(
                model=os.getenv("ANTHROPIC_MODEL", os.getenv("ANTHROPIC_DEV_MODEL")),
                temperature=0.2,
            )

        # ----------------------------
        # Claude (dev-grade)
        # ----------------------------
        if self._provider_enabled_in_capabilities("claude_dev") and os.getenv("ANTHROPIC_API_KEY"):
            provider_map["claude_dev"] = ClaudeClient(
                model=os.getenv("ANTHROPIC_DEV_MODEL", "claude-sonnet-4-5"),
                temperature=0.0,
            )

        # ----------------------------
        # Ollama (local)
        # ----------------------------
        if self._provider_enabled_in_capabilities("ollama_local"):
            provider_map["ollama_local"] = OllamaClient(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
            )

        # ----------------------------
        # Gemini (optional)
        # ----------------------------
        # if self._provider_enabled_in_capabilities("gemini") and os.getenv("GOOGLE_API_KEY"):
        #     provider_map["gemini"] = GeminiClient(
        #         model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        #     )

        return provider_map

    # ----------------------------
    # Public API
    # ----------------------------

    def run(self, task: str) -> Dict[str, Any]:
        started = time.time()

        route = self.router.decide(task)
        plan = self.planner.make_plan(task, route)
        execution: Dict[str, Any]

        final_answer: Optional[str] = None
        judge_info: Optional[Dict[str, Any]] = None
        routing_info: Optional[Dict[str, Any]] = None

        general_mode = self.memory.get_setting("general_mode", "auto")
        if route.strategy == "llm_single" and general_mode != "api_only":
            if route.providers == ["ollama_local"]:
                execution, final_answer, routing_info = self._run_general_local_first(task)
            else:
                execution = self._execute(plan, route)
        elif route.strategy == "llm_single" and general_mode == "api_only":
            api_provider = self._best_available_api_provider()
            if api_provider:
                route = RouteDecision(
                    strategy="llm_single",
                    providers=[api_provider],
                    reason="General mode api_only: using best available API provider.",
                )
                plan = self.planner.make_plan(task, route)
            execution = self._execute(plan, route)
        else:
            execution = self._execute(plan, route)

        if route.strategy in ("llm_single", "llm_multi", "hybrid") and final_answer is None:
            judge_cfg = self.memory.get_judge_config()
            provider_stats = self.memory.get_provider_stats()

            decision, final = self.judge.judge(
                task=task,
                worker_outputs=execution.get("llm", []),
                provider_stats=provider_stats,
                mode=judge_cfg["judge_mode"],
                fixed_provider=judge_cfg["judge_provider"],
            )

            judge_info = {
                "judge_provider": decision.judge_provider,
                "judge_mode": decision.mode,
                "judge_intent": decision.intent,
                "judge_rationale": decision.rationale,
                "judge_score_table": decision.score_table,
            }
            final_answer = final

        evaluation = self._evaluate(route, execution, final_answer)

        if routing_info and routing_info.get("providers_used"):
            route = RouteDecision(
                strategy=route.strategy,
                providers=routing_info["providers_used"],
                reason=route.reason,
            )

        run_record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": task,
            "route": asdict(route),
            "plan": {
                "steps": plan.steps,
                "local_actions": plan.local_actions,
                "prompts": plan.prompts,
            },
            "execution": execution,
            "judge": judge_info,
            "final_answer": final_answer,
            "routing": routing_info,
            "evaluation": evaluation,
            "elapsed_seconds": round(time.time() - started, 3),
            "available_providers": list(self.provider_map.keys()),
        }
        self.memory.add_run(run_record)

        return run_record

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _execute(self, plan: Plan, route: RouteDecision) -> Dict[str, Any]:
        result: Dict[str, Any] = {"local": [], "llm": []}

        if route.strategy == "local_only":
            for action in plan.local_actions:
                tool = action["tool"]
                args = action.get("args", {})
                result["local"].append(self._run_local_tool(tool, args))
            return result

        for provider_name, prompt in plan.prompts.items():
            client = self.provider_map.get(provider_name)
            if not client:
                result["llm"].append(
                    {
                        "provider": provider_name,
                        "success": False,
                        "error": (
                            f"Provider '{provider_name}' not found in provider_map. "
                            f"Available: {list(self.provider_map.keys())}"
                        ),
                    }
                )
                continue

            try:
                text = client.generate(prompt)
                result["llm"].append(
                    {
                        "provider": provider_name,
                        "success": True,
                        "text": text,
                    }
                )
                self.memory.update_provider_stats(provider_name, success=True)
            except Exception as e:
                result["llm"].append(
                    {
                        "provider": provider_name,
                        "success": False,
                        "error": str(e),
                    }
                )
                self.memory.update_provider_stats(provider_name, success=False)

        return result

    def _best_available_api_provider(self) -> str:
        runtime = [p for p in self.provider_map.keys() if p != "ollama_local"]
        if not runtime:
            return ""
        if "openai_dev" in runtime:
            return "openai_dev"
        if "openai" in runtime:
            return "openai"
        if "claude_dev" in runtime:
            return "claude_dev"
        if "claude" in runtime:
            return "claude"
        return runtime[0]

    def _run_general_local_first(self, task: str) -> tuple[Dict[str, Any], str, Dict[str, Any]]:
        execution: Dict[str, Any] = {"local": [], "llm": []}
        routing_info: Dict[str, Any] = {}
        final_answer = ""

        local_client = self.provider_map.get("ollama_local")
        if not local_client:
            api_provider = self._best_available_api_provider()
            if api_provider:
                prompt = f"Task:\n{task}\n\nBe direct and practical."
                text = self.provider_map[api_provider].generate(prompt)
                execution["llm"].append({"provider": api_provider, "success": True, "text": text})
                routing_info = {"route": "api", "reason": "local_unavailable", "confidence": 0.0}
                return execution, text, routing_info
            return execution, "Local model unavailable and no API providers available.", {
                "route": "none",
                "reason": "no_providers",
                "confidence": 0.0,
            }

        threshold = float(self.memory.get_setting("general_confidence_threshold", 0.90) or 0.90)
        web_first_threshold = float(self.memory.get_setting("web_first_threshold", 0.70) or 0.70)
        general_mode = self.memory.get_setting("general_mode", "auto")
        intent = classify_intent(task)
        stats = self.memory.get_general_routing_stats()
        learned_conf = float(stats.get("per_intent", {}).get(intent, {}).get("confidence", 0.0) or 0.0)

        local_prompt = build_local_assessment_prompt(task, threshold)
        local_raw = local_client.generate(local_prompt)
        assessment, error = parse_local_assessment(local_raw)

        invalid_json = assessment is None
        if assessment is None:
            factual = is_factual_query(task) or is_authoritative_query(task)
            assessment = {
                "final_answer": local_raw.strip(),
                "confidence": 0.0,
                "escalate_to": "web" if factual else "api",
                "search_query": "",
                "uncertainty_reasons": ["invalid_json"],
                "suggested_search_query": "",
            }

        if general_mode == "local_only":
            decision = {"route": "local", "reason": "local_only", "confidence": float(assessment.get("confidence", 0.0) or 0.0)}
        else:
            decision = decide_general_route(
                task=task,
                assessment=assessment,
                confidence_threshold=threshold,
                web_first_threshold=web_first_threshold,
                learned_confidence=learned_conf,
            )

        route = decision["route"]
        confidence = decision["confidence"]
        reason = decision["reason"]
        query = assessment.get("search_query") or assessment.get("suggested_search_query") or task
        used_web = False
        used_api = False
        local_only_answer = False

        providers_used = ["ollama_local"]

        if route == "local":
            final_answer = assessment.get("final_answer", "")
            if general_mode == "local_only":
                uncertainty = assessment.get("uncertainty_reasons") or []
                if uncertainty or confidence < threshold:
                    final_answer = (
                        f"{final_answer}\n\n"
                        "Note: Uncertain response from local model."
                    )
            local_only_answer = True
            execution["llm"].append({"provider": "ollama_local", "success": True, "text": final_answer})
            routing_info = {
                "route": "local",
                "confidence": confidence,
                "reason": reason,
                "providers_used": providers_used,
            }
            print(f"Route: local (conf={confidence:.2f})")
        elif route == "web":
            used_web = True
            authoritative = is_authoritative_query(task)
            results = []
            status = "error"
            reason_detail = ""
            search_tool = ""

            def _low_quality(items: list[dict]) -> bool:
                if not items:
                    return True
                has_url = any(bool(i.get("url")) for i in items)
                has_snippet = any(bool(i.get("snippet")) for i in items)
                return not (has_url and has_snippet)

            def _log_search(tool_name: str, tool_status: str, detail: str, count: int) -> None:
                if tool_status == "ok":
                    print(f"WebSearch: {tool_name} query=\"{query}\" results={count}")
                elif tool_status in ("empty_results", "empty"):
                    print(f"WebSearch: {tool_name} EMPTY")
                else:
                    print(f"WebSearch: {tool_name} FAILED reason=\"{detail}\"")

            def _record_search(tool_name: str, tool_status: str, items: list[dict], detail: str) -> None:
                providers_used.append(f"web_search_{tool_name}")
                execution["local"].append(
                    {
                        "tool": f"web_search_{tool_name}",
                        "args": {"query": query},
                        "success": tool_status == "ok",
                        "output": items,
                        "error": "" if tool_status == "ok" else detail,
                    }
                )

            if authoritative:
                results, status, reason_detail = web_search_vertex_status(query, num_results=3)
                search_tool = "vertex_search"
                _record_search(search_tool, status, results, reason_detail)
                _log_search("vertex_search", status, reason_detail, len(results))
                if status != "ok" or _low_quality(results):
                    open_results = web_search(query, max_results=3)
                    open_status = "ok" if open_results else "empty_results"
                    _record_search("open_web", open_status, open_results, "empty_results")
                    _log_search("open_web", open_status, "empty_results", len(open_results))
                    if open_results:
                        results = open_results
                        status = "ok"
                        search_tool = "open_web"
            else:
                open_results = web_search(query, max_results=3)
                open_status = "ok" if open_results else "empty_results"
                _record_search("open_web", open_status, open_results, "empty_results")
                _log_search("open_web", open_status, "empty_results", len(open_results))
                results = open_results
                status = open_status
                search_tool = "open_web"
                if status != "ok" or _low_quality(results):
                    vertex_results, vertex_status, vertex_detail = web_search_vertex_status(query, num_results=3)
                    _record_search("vertex_search", vertex_status, vertex_results, vertex_detail)
                    _log_search("vertex_search", vertex_status, vertex_detail, len(vertex_results))
                    if vertex_status == "ok":
                        results = vertex_results
                        status = "ok"
                        search_tool = "vertex_search"

            evidence = summarize_evidence(results)
            web_weak = len(results) < 1
            if web_weak:
                api_provider = self._best_available_api_provider()
                if api_provider:
                    used_api = True
                    providers_used.append(api_provider)
                    api_prompt = (
                        "You are a reliable assistant. Use the web evidence below to answer the task.\n"
                        "If evidence is weak or conflicting, state uncertainty.\n\n"
                        f"Task:\n{task}\n\n"
                        f"Web evidence:\n{evidence}\n"
                    )
                    text = self.provider_map[api_provider].generate(api_prompt)
                    execution["llm"].append({"provider": api_provider, "success": True, "text": text})
                    final_answer = text
                    print(f"Route: local->web->api (conf={confidence:.2f})")
                else:
                    final_answer = "Web search returned no results and no API provider is available."
                    print(f"Route: local->web ({search_tool})")
            else:
                synth_prompt = build_local_evidence_prompt(task, evidence)
                final_answer = local_client.generate(synth_prompt)
                execution["llm"].append({"provider": "ollama_local", "success": True, "text": final_answer})
                if not any(r.get("url") and r.get("url") in final_answer for r in results):
                    if results and results[0].get("url"):
                        final_answer = f"{final_answer}\n\nSource: {results[0]['url']}"
                print(f"Route: local->web ({search_tool})")

            routing_info = {
                "route": "local->web->api" if used_api else "local->web",
                "confidence": confidence,
                "reason": reason,
                "query": query,
                "search_tool": search_tool,
                "providers_used": providers_used,
            }
        elif route == "api":
            api_provider = self._best_available_api_provider()
            if api_provider:
                used_api = True
                providers_used.append(api_provider)
                api_prompt = f"Task:\n{task}\n\nBe direct and practical."
                text = self.provider_map[api_provider].generate(api_prompt)
                execution["llm"].append({"provider": api_provider, "success": True, "text": text})
                final_answer = text
                print(f"Route: local->api (conf={confidence:.2f})")
                routing_info = {
                    "route": "local->api",
                    "confidence": confidence,
                    "reason": reason,
                    "providers_used": providers_used,
                }
            else:
                final_answer = assessment.get("final_answer", "")
                routing_info = {
                    "route": "local",
                    "confidence": confidence,
                    "reason": "api_unavailable",
                    "providers_used": providers_used,
                }
        else:
            final_answer = assessment.get("final_answer", "")
            routing_info = {
                "route": "local",
                "confidence": confidence,
                "reason": reason,
                "providers_used": providers_used,
            }

        if invalid_json and general_mode == "local_only":
            final_answer = (
                f"{assessment.get('final_answer', '')}\n\n"
                "Note: Local model did not provide structured confidence; treat this as uncertain."
            )

        self.memory.update_general_routing_stats(
            info={
                "intent": intent,
                "local_json_valid": not invalid_json,
                "local_only_answer": local_only_answer,
                "web_escalated": used_web,
                "api_escalated": used_api,
                "invalid_json": invalid_json,
            }
        )

        return execution, final_answer, routing_info

    def _run_local_tool(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if tool == "list_dir":
                path = args.get("path", ".")
                items = list_dir(path)
                return {"tool": tool, "args": args, "success": True, "output": items}

            if tool == "read_file":
                path = args["path"]
                content = read_file(path)
                return {"tool": tool, "args": args, "success": True, "output": content}

            if tool == "write_file":
                path = args["path"]
                content = args["content"]
                write_file(path, content)
                return {"tool": tool, "args": args, "success": True, "output": f"Wrote {len(content)} chars."}

            return {"tool": tool, "args": args, "success": False, "error": "Unknown tool name."}

        except Exception as e:
            return {"tool": tool, "args": args, "success": False, "error": str(e)}

    def _evaluate(self, route: RouteDecision, execution: Dict[str, Any], final_answer: Any) -> Dict[str, Any]:
        if route.strategy == "local_only":
            ok = all(step.get("success") for step in execution.get("local", []))
            return {"success": ok, "notes": "Local tool execution success check."}

        ok = final_answer is not None and isinstance(final_answer, str) and len(final_answer.strip()) > 0
        return {"success": ok, "notes": "Judge produced a final answer."}

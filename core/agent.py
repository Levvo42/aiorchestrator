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

# Local tools
from tools.local_exec import read_file, write_file, list_dir

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
        execution = self._execute(plan, route)

        final_answer: Optional[str] = None
        judge_info: Optional[Dict[str, Any]] = None

        if route.strategy in ("llm_single", "llm_multi", "hybrid"):
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

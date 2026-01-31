"""
main.py
-------
Entry point for your AI Orchestrator.

You can type tasks, or commands like:
- Set Judge: openai
- Set Judge: gemini
- Set Judge Mode: auto
- Set Judge Mode: fixed
- Show Judge
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import json
from pathlib import Path
from typing import Optional
from core.agent import Agent
from core.memory import MemoryStore
from dev.dev_command import run_dev_request, apply_dev_patch


def load_capabilities(path: str = "core/capabilities.json") -> dict:
    """Load capabilities registry from disk."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def normalize_provider_name(name: str) -> str:
    """
    Normalize user-friendly input to internal provider keys.
    Example: 'OpenAI' -> 'openai'
    """
    return name.strip().lower()


def handle_command(text: str, memory: MemoryStore) -> Optional[str]:
    """
    Handle console commands that change judge settings.
    Returns a user-friendly message if a command was handled, otherwise None.
    """
    t = text.strip()
    if t.lower() == "show dev settings":
        cfg = memory.state.get("settings", {})
        lines = ["Dev settings:"]
        keys = [
            "dev_mode", "dev_authors", "dev_judge_provider",
            "dev_min_authors", "dev_max_authors", "dev_exploration_rate"
        ]
        for k in keys:
            lines.append(f"- {k}: {cfg.get(k)}")
        return "\n".join(lines)

    if t.lower().startswith("set dev mode:"):
        mode = t.split(":", 1)[1].strip().lower()
        if mode not in ("auto", "fixed"):
            return "Invalid. Use: Set Dev Mode: auto | fixed"
        memory.set_setting("dev_mode", mode)
        return f"Dev mode set to: {mode}"

    if t.lower().startswith("set dev judge:"):
        provider = t.split(":", 1)[1].strip().lower()
        memory.set_setting("dev_judge_provider", provider)
        return f"Dev judge provider set to: {provider}"

    if t.lower().startswith("set dev authors:"):
        raw = t.split(":", 1)[1].strip()
        # Accept comma-separated list
        authors = [a.strip().lower() for a in raw.split(",") if a.strip()]
        memory.set_setting("dev_authors", authors if authors else None)
        return f"Dev authors set to: {authors}"

    if t.lower() == "help":
        return (
            "Available commands:\n"
            "- Help\n"
            "- Show Settings\n"
            "- Show Judge\n"
            "- Set Judge: <provider>\n"
            "- Set Judge Mode: auto | fixed\n"
            "- Set Verbosity: full | normal | final\n"
            "\nExamples:\n"
            "  Set Judge: gemini\n"
            "  Set Verbosity: final\n"
            "  Dev: < request >\n"

            " Show Dev Settings \n"
            " Set Dev Mode: auto | fixed \n"
            " Set Dev Judge: < provider > \n"
            " Set Dev Authors: a, b, c \n"
        )

    # Show current judge configuration
    if t.lower() == "show judge":
        cfg = memory.get_judge_config()
        return f"Judge mode: {cfg['judge_mode']}, Judge provider: {cfg['judge_provider']}"

    # Set Judge Mode: auto/fixed
    if t.lower().startswith("set judge mode:"):
        mode = t.split(":", 1)[1].strip().lower()
        if mode not in ("auto", "fixed"):
            return "Invalid judge mode. Use: Set Judge Mode: auto  OR  Set Judge Mode: fixed"

        memory.set_setting("judge_mode", mode)

        # If switching to auto, we can clear fixed provider to avoid confusion
        if mode == "auto":
            memory.set_setting("judge_provider", None)

        return f"Judge mode set to: {mode}"

    # Set Judge: provider_name (puts mode into fixed)
    if t.lower().startswith("set judge:"):
        provider = normalize_provider_name(t.split(":", 1)[1])
        if not provider:
            return "Usage: Set Judge: openai  OR  Set Judge: gemini"

        memory.set_setting("judge_mode", "fixed")
        memory.set_setting("judge_provider", provider)
        return f"Judge set to: {provider} (mode=fixed)"

    # Friendly alternative: "Judge with OpenAI"
    if t.lower().startswith("judge with "):
        provider = normalize_provider_name(t[len("judge with "):])
        if not provider:
            return "Usage: Judge with openai  OR  Judge with gemini"

        memory.set_setting("judge_mode", "fixed")
        memory.set_setting("judge_provider", provider)
        return f"Judge set to: {provider} (mode=fixed)"

    # Show all settings
    if t.lower() == "show settings":
        cfg = memory.state.get("settings", {})
        lines = ["Current settings:"]
        for k, v in cfg.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    # Set Verbosity
    if t.lower().startswith("set verbosity:"):
        level = t.split(":", 1)[1].strip().lower()
        if level not in ("full", "normal", "final"):
            return "Invalid verbosity. Use: Set Verbosity: full | normal | final"

        memory.set_verbosity(level)
        return f"Verbosity set to: {level}"


    return None


def print_run_summary(run: dict, verbosity: str) -> None:
    """
    Print output based on verbosity.

    verbosity modes:
    - "final": print ONLY the final answer (best for normal use)
    - "normal": print route + judge + final answer (minimal insight)
    - "full": print everything (debug mode)
    """

    # Safety: if verbosity is unknown, treat it as "normal"
    if verbosity not in ("final", "normal", "full"):
        verbosity = "normal"

    # 1) FINAL ONLY
    if verbosity == "final":
        final = run.get("final_answer")
        if final:
            print(final)
        else:
            # If it's a local_only run, final_answer may be None, so show local output.
            local = run.get("execution", {}).get("local", [])
            if local:
                print(local)
            else:
                print("No final answer available.")
        print()
        return

    # 2) NORMAL (route + judge + final)
    route = run.get("route", {})
    print("\n=== ROUTE ===")
    print(f"Strategy: {route.get('strategy')}")
    print(f"Providers: {route.get('providers')}")
    print(f"Reason: {route.get('reason')}")

    judge = run.get("judge")
    if judge:
        print("\n=== JUDGE ===")
        print(f"Judge provider: {judge.get('judge_provider')}")
        print(f"Judge mode: {judge.get('judge_mode')}")
        print(f"Judge intent: {judge.get('judge_intent')}")
        # Only print score table in FULL mode, because it's noisy
        if verbosity == "full" and judge.get("judge_mode") == "auto":
            print(f"Score table: {judge.get('judge_score_table')}")

    final = run.get("final_answer")
    if final:
        print("\n=== FINAL ANSWER ===")
        print(final)

    # If this was local_only, show local output in normal mode too
    local = run.get("execution", {}).get("local", [])
    if local:
        print("\n=== LOCAL OUTPUT ===")
        for item in local:
            if item.get("success"):
                print(f"- {item.get('tool')} OK")
                print(item.get("output"))
            else:
                print(f"- {item.get('tool')} FAILED: {item.get('error')}")

    # 3) FULL (also show plan + worker outputs + evaluation details)
    if verbosity == "full":
        plan = run.get("plan", {})
        steps = plan.get("steps", [])
        prompts = plan.get("prompts", {})

        print("\n=== PLAN ===")
        for i, step in enumerate(steps, start=1):
            print(f"{i}. {step}")

        # Worker outputs can be long, but in FULL mode we show them
        llm = run.get("execution", {}).get("llm", [])
        if llm:
            print("\n=== WORKER OUTPUTS ===")
            for item in llm:
                provider = item.get("provider")
                if item.get("success"):
                    print(f"\n--- {provider} ---\n{item.get('text')}\n")
                else:
                    print(f"\n--- {provider} FAILED ---\n{item.get('error')}\n")

        evaluation = run.get("evaluation", {})
        print("\n=== EVALUATION ===")
        print(f"Success: {evaluation.get('success')}")
        print(f"Notes: {evaluation.get('notes')}")
        print(f"Elapsed: {run.get('elapsed_seconds')}s")

    print()



if __name__ == "__main__":
    load_dotenv()

    capabilities = load_capabilities()
    memory = MemoryStore(state_path="memory/state.local.json", seed_path="memory/state.json")
    agent = Agent(capabilities=capabilities, memory=memory)
    # Holds a dev report that has been proposed but not yet confirmed/applied.
    pending_dev_report = None


    print("AI-Orchestrator v0.2 (with Judge)")
    print("Commands: Show Judge | Set Judge: openai | Set Judge Mode: auto/fixed")
    print("Type a task and press Enter. Empty input quits.\n")
    print("DEBUG providers:", list(agent.provider_map.keys()))

    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            # Happens if the run console doesn't provide stdin (or closes it)
            print("\nNo interactive input available (EOF). Check PyCharm run config: enable 'Emulate terminal'.")
            break

        if not text:
            # If we're waiting for a yes/no on a dev patch, don't exit on empty input.
            if pending_dev_report is not None:
                print("Please answer: yes or no")
                continue
            break

        # ---------------------------------------------------------
        # A) PENDING DEV CONFIRMATION STATE (YES/NO consumes input)
        # ---------------------------------------------------------
        if pending_dev_report is not None:
            answer = text.strip().lower()

            if answer in ("y", "yes"):
                pending_dev_report = apply_dev_patch(repo_root=".", report=pending_dev_report)

                print("\n=== APPLY RESULT ===")
                print(f"Applied: {pending_dev_report['apply']['applied']}")
                if pending_dev_report["apply"]["error"]:
                    print(f"Error: {pending_dev_report['apply']['error']}")
                else:
                    print(f"Changed files: {pending_dev_report['apply']['changed_files']}")
                    print(f"Validation OK: {pending_dev_report['apply']['validation_ok']}")
                    print(f"Validation output:\n{pending_dev_report['apply']['validation_output']}")

                # Store dev run in memory
                memory.add_run({
                    "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
                    "task": f"DEV: {pending_dev_report.get('request', '')}",
                    "dev_report": pending_dev_report
                })

                # Clear pending state after handling
                pending_dev_report = None
                continue

            if answer in ("n", "no"):
                print("Patch not applied.")
                pending_dev_report = None
                continue

            # If they typed something else, keep waiting for a valid yes/no
            print("Please answer: yes or no")
            continue

        # -------------------
        # B) NORMAL COMMANDS
        # -------------------
        msg = handle_command(text, memory)
        if msg:
            print(msg)
            continue

        # ------------------------------------
        # C) DEV COMMAND (creates pending state)
        # ------------------------------------
        if text.lower().startswith("dev:"):
            dev_request = text.split(":", 1)[1].strip()
            if not dev_request:
                print("Usage: Dev: <describe the change you want>")
                continue

            report = run_dev_request(
                repo_root=".",
                request=dev_request,
                capabilities=capabilities,
                memory=memory,
                provider_map=agent.provider_map
            )

            print("\n=== DEV POLICY ===")
            print(f"Mode: {report['policy']['mode']}")
            print(f"Authors: {report['policy'].get('author_providers', [])}")
            print(f"Judge: {report['policy'].get('judge_provider')}")
            print(f"Reason: {report['policy']['reason']}")

            print("\n=== DEV JUDGE RATIONALE ===")
            print(report["judge"]["rationale"] or "(no rationale)")

            print("\n=== PROPOSED PATCH ===")
            print(report["chosen_patch"] or "(no patch produced)")

            # IMPORTANT: we DO NOT call input("Apply patch?") here anymore.
            # Instead, we set pending state and let the next user input be the answer.
            pending_dev_report = report
            print("\nApply patch? (yes/no):")
            continue

        # --------------------
        # D) NORMAL TASK ROUTE
        # --------------------
        run = agent.run(text)
        verbosity = memory.get_verbosity()
        print_run_summary(run, verbosity)



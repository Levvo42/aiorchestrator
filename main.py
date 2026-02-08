# python
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
import subprocess
import json
from pathlib import Path
from typing import Optional
from core.agent import Agent
from core.memory import MemoryStore
from core.env import validate_required_env_vars
from dev.dev_command import run_dev_request, run_dev_fix_request, apply_dev_patch


def load_capabilities(path: str = "core/capabilities.json") -> dict:
    """Load capabilities registry from disk."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _validate_env_for_enabled_providers(capabilities: dict) -> None:
    providers = capabilities.get("providers", {})
    required = []
    for _name, cfg in providers.items():
        if cfg.get("enabled", False) and cfg.get("env_key_required"):
            required.append(cfg["env_key_required"])
    if required:
        validate_required_env_vars(required)


def normalize_provider_name(name: str) -> str:
    """
    Normalize user-friendly input to internal provider keys.
    Example: 'OpenAI' -> 'openai'
    """
    return name.strip().lower()


def _show_dev_progress(step: str, info: Optional[str] = None) -> None:
    """Lightweight progress printer for dev self-patch flow."""
    if info:
        print(f"[progress] {step}: {info}")
    else:
        print(f"[progress] {step}")


def handle_command(text: str, memory: MemoryStore,
                   pending_dev_report: Optional[dict] = None,
                   last_applied_dev_report: Optional[dict] = None) -> Optional[str]:
    """
    Handle console commands that change judge settings.
    Returns a user-friendly message if a command was handled, otherwise None.

    Note: accepts pending and last-applied dev report to support 'Dev: Commit with message' command.
    """
    t = text.strip()
    if t.lower() == "show dev settings":
        cfg = memory.state.get("settings", {})
        lines = ["Dev settings:"]
        keys = [
            "dev_mode", "dev_authors", "dev_judge_provider",
            "dev_min_authors", "dev_max_authors", "dev_exploration_rate",
            "dev_judge_mode"
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

    if t.lower().startswith("set dev judge mode:"):
        mode = t.split(":", 1)[1].strip().lower()
        if mode not in ("auto", "local_only", "api_only"):
            return "Invalid. Use: Set Dev Judge Mode: auto | local_only | api_only"
        memory.set_setting("dev_judge_mode", mode)
        return f"Dev judge mode set to: {mode}"

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

    if t.lower().startswith("dev: commit with message"):
        # Extract message after "Dev: Commit with message"
        parts = text.split("\"")
        if len(parts) < 2:
            return "Usage: Dev: Commit with message \"<your commit message>\""

        commit_msg = parts[1].strip()
        if not commit_msg:
            return "Commit message cannot be empty."

        # Use last_applied_dev_report (the applied patch) for committing
        if last_applied_dev_report is None:
            return "No dev patch has been applied. Use 'Dev: <request>' first, then apply the patch."

        if not last_applied_dev_report.get("apply", {}).get("applied"):
            return "Last dev patch was not applied successfully."

        if not last_applied_dev_report.get("apply", {}).get("validation_ok"):
            return "Cannot commit: last dev patch did not pass validation."

        # Check if working tree is clean (only staged changes allowed)
        try:
            result = subprocess.run(["git", "diff", "--quiet"], cwd=".", check=False)
            if result.returncode != 0:
                return "Cannot commit: working tree has unstaged changes. Stage your changes first with 'git add'."

            subprocess.run(["git", "commit", "-m", commit_msg], cwd=".", check=True, capture_output=True, text=True)
            return f"Committed successfully with message: \"{commit_msg}\""
        except subprocess.CalledProcessError as e:
            return f"Git commit failed: {e.stderr if e.stderr else str(e)}"

    if t.lower() == "help":
        return (
            "Available commands:\n"
            "General:\n"
            "  Help\n"
            "  Show Settings\n"
            "  Show Judge\n"
            "  Set Judge: <provider>\n"
            "  Set Judge Mode: auto | fixed | local_only | api_only\n"
            "  Set Judge Threshold: <0..1>\n"
            "  Set General Mode: auto | local_only | api_only\n"
            "  Set General Confidence Threshold: <0..1>\n"
            "  Set Web First Threshold: <0..1>\n"
            "  Set Verbosity: full | normal | final\n"
            "\nDev workflow:\n"
            "  Dev: <request>\n"
            "  Show Dev Settings\n"
            "  Set Dev Mode: auto | fixed\n"
            "  Set Dev Judge: <provider>\n"
            "  Set Dev Judge Mode: auto | local_only | api_only\n"
            "  Set Dev Authors: a, b, c\n"
            "  Dev: Commit with message \"<msg>\"\n"
            "\nExamples:\n"
            "  Set Judge: gemini\n"
            "  Set Verbosity: final\n"
            "  Dev: add logging to router\n"
        )

    # Show current judge configuration
    if t.lower() == "show judge":
        cfg = memory.get_judge_config()
        return f"Judge mode: {cfg['judge_mode']}, Judge provider: {cfg['judge_provider']}"

    # Set Judge Mode: auto/fixed
    if t.lower().startswith("set judge mode:"):
        mode = t.split(":", 1)[1].strip().lower()
        if mode not in ("auto", "fixed", "local_only", "api_only"):
            return "Invalid judge mode. Use: Set Judge Mode: auto | fixed | local_only | api_only"

        memory.set_setting("judge_mode", mode)

        # If switching to auto, we can clear fixed provider to avoid confusion
        if mode == "auto":
            memory.set_setting("judge_provider", None)

        return f"Judge mode set to: {mode}"

    if t.lower().startswith("set judge threshold:"):
        raw = t.split(":", 1)[1].strip()
        try:
            value = float(raw)
        except ValueError:
            return "Invalid judge threshold. Use a number between 0 and 1."
        if value < 0.0 or value > 1.0:
            return "Invalid judge threshold. Use a number between 0 and 1."
        memory.set_setting("judge_threshold", value)
        return f"Judge threshold set to: {value}"

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

    if t.lower().startswith("set general mode:"):
        mode = t.split(":", 1)[1].strip().lower()
        if mode not in ("auto", "local_only", "api_only"):
            return "Invalid general mode. Use: Set General Mode: auto | local_only | api_only"
        memory.set_setting("general_mode", mode)
        return f"General mode set to: {mode}"

    if t.lower().startswith("set general confidence threshold:"):
        raw = t.split(":", 1)[1].strip()
        try:
            value = float(raw)
        except ValueError:
            return "Invalid general confidence threshold. Use a number between 0 and 1."
        if value < 0.0 or value > 1.0:
            return "Invalid general confidence threshold. Use a number between 0 and 1."
        memory.set_setting("general_confidence_threshold", value)
        return f"General confidence threshold set to: {value}"

    if t.lower().startswith("set web first threshold:"):
        raw = t.split(":", 1)[1].strip()
        try:
            value = float(raw)
        except ValueError:
            return "Invalid web first threshold. Use a number between 0 and 1."
        if value < 0.0 or value > 1.0:
            return "Invalid web first threshold. Use a number between 0 and 1."
        memory.set_setting("web_first_threshold", value)
        return f"Web first threshold set to: {value}"

    # Set Verbosity
    if t.lower().startswith("set verbosity:"):
        level = t.split(":", 1)[1].strip().lower()
        if level not in ("full", "normal", "final"):
            return "Invalid verbosity. Use: Set Verbosity: full | normal | final"

        memory.set_verbosity(level)
        return f"Verbosity set to: {level}"

    return None


def _safe_input(prompt: str) -> Optional[str]:
    try:
        return input(prompt)
    except EOFError:
        print("\nNo interactive input available (EOF).")
        return None


def _prompt_yes_no(prompt: str) -> bool:
    while True:
        ans = _safe_input(prompt)
        if ans is None:
            return False
        a = ans.strip().lower()
        if a in ("y", "yes"):
            return True
        if a in ("n", "no"):
            return False
        print("Please answer: yes or no")


def _run_git(repo_root: str, args):
    try:
        completed = subprocess.run(
            ["git"] + args,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        return False, str(e)
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout or "").strip()
    return True, (completed.stdout or "").strip()


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
    _validate_env_for_enabled_providers(capabilities)
    memory = MemoryStore("memory/state.json")
    agent = Agent(capabilities=capabilities, memory=memory)
    # Holds a dev report that has been proposed but not yet confirmed/applied.
    pending_dev_report = None
    last_applied_dev_report = None
    pending_dev_invalid_shown = False


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
                pending_dev_invalid_shown = False
                # Store the report before applying so we can reference it for commits
                last_applied_dev_report = pending_dev_report

                _show_dev_progress("apply", "Starting apply of proposed patch")
                pending_dev_report = apply_dev_patch(repo_root=".", report=pending_dev_report)

                _show_dev_progress("apply", "Apply completed")

                print("\n=== APPLY RESULT ===")
                print(f"Applied: {pending_dev_report['apply']['applied']}")
                if pending_dev_report["apply"].get("error"):
                    print(f"Error: {pending_dev_report['apply']['error']}")
                else:
                    print(f"Changed files: {pending_dev_report['apply'].get('changed_files')}")
                    print(f"Validation OK: {pending_dev_report['apply'].get('validation_ok')}")
                    print(f"Validation output:\n{pending_dev_report['apply'].get('validation_output')}")

                # ✅ If apply failed, offer auto-fix + re-propose BEFORE doing anything else.
                if not pending_dev_report["apply"].get("applied"):
                    fix_depth = int((pending_dev_report.get("fix", {}) or {}).get("depth", 0) or 0)
                    if fix_depth >= 1:
                        print("Auto-fix already attempted for this request; not repeating.")
                    elif _prompt_yes_no("Attempt auto-fix and re-propose a patch? (yes/no) "):
                        _show_dev_progress("fix", "Requesting auto-fix proposal")
                        fix_report = run_dev_fix_request(
                            repo_root=".",
                            failed_report=pending_dev_report,
                            capabilities=capabilities,
                            memory=memory,
                            provider_map=agent.provider_map,
                        )
                        _show_dev_progress("fix", "Auto-fix proposal received")

                        print("\n=== PROPOSED FIX PATCH ===")
                        print(fix_report.get("chosen_patch") or "(no patch produced)")

                        # Replace pending report with the new proposal and go back to yes/no apply.
                        pending_dev_report = fix_report
                        print("\nApply patch? (yes/no):")
                        continue

                # Update provider stats ONLY after an explicit apply confirmation.
                # This keeps the proposal step side-effect free and reproducible.
                try:
                    for a in pending_dev_report.get("authors", []):
                        provider = a.get("provider")
                        success = bool(a.get("success"))
                        if provider:
                            memory.update_provider_stats(provider, success=success)

                    j = pending_dev_report.get("judge", {})
                    j_provider = j.get("provider")
                    j_success = bool(j.get("success"))
                    if j_provider:
                        memory.update_provider_stats(j_provider, success=j_success)
                except Exception:
                    # Stats should never break the dev flow.
                    pass

                try:
                    local_info = pending_dev_report.get("local_judge", {}) or {}
                    apply_result = pending_dev_report.get("apply", {}) or {}
                    memory.update_local_judge_stats(
                        info={
                            "intent": local_info.get("intent"),
                            "local_provider": local_info.get("local_provider"),
                            "api_provider": local_info.get("api_provider"),
                            "local_attempted": (local_info.get("local") or {}).get("attempted"),
                            "local_valid_json": (local_info.get("local") or {}).get("valid_json"),
                            "escalated": (local_info.get("local") or {}).get("escalated"),
                            "local_patch_index": (local_info.get("local") or {}).get("patch_index"),
                            "api_patch_index": (local_info.get("api") or {}).get("patch_index"),
                            "selected_source": local_info.get("selected_source"),
                        },
                        apply_result=apply_result,
                    )
                except Exception:
                    pass

                # Store dev run in memory
                memory.add_run({
                    "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
                    "task": f"DEV: {pending_dev_report.get('request', '')}",
                    "dev_report": pending_dev_report
                })

                # Update last_applied_dev_report with the full apply results
                last_applied_dev_report = pending_dev_report

                # Offer commit + push after successful apply + validation
                if pending_dev_report["apply"].get("applied") and pending_dev_report["apply"].get("validation_ok"):
                    if _prompt_yes_no("Commit and push to GitHub? (yes/no) "):
                        commit_msg = _safe_input("Commit message: ")
                        if commit_msg is None:
                            pass
                        else:
                            commit_msg = commit_msg.strip()
                            if not commit_msg:
                                print("Commit message cannot be empty.")
                            else:
                                changed_files = pending_dev_report["apply"].get("changed_files", []) or []
                                if not changed_files:
                                    print("No changed files to commit.")
                                else:
                                    ok, out = _run_git(".", ["add", "--"] + changed_files)
                                    if not ok:
                                        print(f"Git add failed: {out}")
                                    else:
                                        ok, out = _run_git(".", ["commit", "-m", commit_msg])
                                        if not ok:
                                            print(f"Git commit failed: {out}")
                                        else:
                                            ok, out = _run_git(".", ["show", "--name-only", "--stat", "HEAD"])
                                            if not ok:
                                                print(f"Git show failed: {out}")
                                            else:
                                                if out:
                                                    print(out)
                                                if _prompt_yes_no("Push now? (yes/no) "):
                                                    ok, out = _run_git(".", ["push"])
                                                    if not ok:
                                                        print(f"Git push failed: {out}")
                                                    elif out:
                                                        print(out)

                pending_dev_report = None
                continue

            if answer in ("n", "no"):
                pending_dev_invalid_shown = False
                print("Patch not applied.")
                pending_dev_report = None
                continue

            # If they typed something else, keep waiting for a valid yes/no
            if not pending_dev_invalid_shown and answer:
                print("Please answer: yes or no")
                pending_dev_invalid_shown = True
            continue

        # -------------------
        # B) NORMAL COMMANDS
        # -------------------
        msg = handle_command(text, memory, pending_dev_report, last_applied_dev_report)
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

            _show_dev_progress("request", "Requesting patch proposal")
            report = run_dev_request(
                repo_root=".",
                request=dev_request,
                capabilities=capabilities,
                memory=memory,
                provider_map=agent.provider_map
            )
            _show_dev_progress("request", "Patch proposal received")

            print("\n=== DEV POLICY ===")
            print(f"Mode: {report['policy']['mode']}")
            print(f"Authors: {report['policy']['authors']}")
            print(f"Judge: {report['policy']['judge']}")
            print(f"Reason: {report['policy']['reason']}")

            print("\n=== DEV JUDGE RATIONALE ===")
            print(report["judge"]["rationale"] or "(no rationale)")

            print("\n=== PROPOSED PATCH ===")
            print(report["chosen_patch"] or "(no patch produced)")

            pending_dev_report = report
            pending_dev_invalid_shown = False
            print("\nApply patch? (yes/no):")
            continue

        # --------------------
        # D) NORMAL TASK ROUTE
        # --------------------
        run = agent.run(text)
        verbosity = memory.get_verbosity()
        print_run_summary(run, verbosity)

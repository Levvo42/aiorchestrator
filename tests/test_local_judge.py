import tempfile
import unittest
from pathlib import Path

from core.memory import MemoryStore
from dev.dev_command import _evaluate_local_judge_output


def _sample_patch(path: str = "foo.txt") -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )


class LocalJudgeTests(unittest.TestCase):
    def test_invalid_json_triggers_escalation(self) -> None:
        result = _evaluate_local_judge_output(
            raw_output="not json",
            candidate_patch_texts=[_sample_patch()],
            threshold=0.90,
            learned_confidence=1.0,
        )
        self.assertTrue(result["escalate"])
        self.assertEqual(result["escalation_reason"], "invalid_json")

    def test_confidence_below_threshold_triggers_escalation(self) -> None:
        raw = (
            '{"patch_index": 0, "confidence": 0.5, "uncertainty_reasons": [], "rationale": "low"}'
        )
        result = _evaluate_local_judge_output(
            raw_output=raw,
            candidate_patch_texts=[_sample_patch()],
            threshold=0.90,
            learned_confidence=1.0,
        )
        self.assertTrue(result["escalate"])
        self.assertEqual(result["escalation_reason"], "confidence_below_threshold")

    def test_uncertainty_reasons_trigger_escalation(self) -> None:
        raw = (
            '{"patch_index": 0, "confidence": 1.0, "uncertainty_reasons": ["risk"], "rationale": "uncertain"}'
        )
        result = _evaluate_local_judge_output(
            raw_output=raw,
            candidate_patch_texts=[_sample_patch()],
            threshold=0.90,
            learned_confidence=1.0,
        )
        self.assertTrue(result["escalate"])
        self.assertEqual(result["escalation_reason"], "uncertainty_reasons")

    def test_stats_persistence_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.local.json"
            seed_path = Path(tmp) / "state.json"
            memory = MemoryStore(state_path=str(state_path), seed_path=str(seed_path))

            memory.update_local_judge_stats(
                info={
                    "intent": "general_judge",
                    "local_provider": "ollama_local",
                    "api_provider": "openai_dev",
                    "local_attempted": True,
                    "local_valid_json": True,
                    "escalated": True,
                    "local_patch_index": 0,
                    "api_patch_index": 0,
                    "selected_source": "api",
                },
                apply_result={"applied": True, "validation_ok": True},
            )

            memory_reloaded = MemoryStore(state_path=str(state_path), seed_path=str(seed_path))
            stats = memory_reloaded.get_local_judge_stats()
            overall = stats.get("overall", {})
            self.assertEqual(overall.get("local_decisions"), 1)
            self.assertEqual(overall.get("local_vs_api_agreement"), 1)

    def test_confidence_update_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.local.json"
            seed_path = Path(tmp) / "state.json"
            memory = MemoryStore(state_path=str(state_path), seed_path=str(seed_path))

            memory.update_local_judge_stats(
                info={
                    "intent": "general_judge",
                    "local_provider": "ollama_local",
                    "api_provider": "openai_dev",
                    "local_attempted": True,
                    "local_valid_json": True,
                    "escalated": True,
                    "local_patch_index": 1,
                    "api_patch_index": 1,
                    "selected_source": "api",
                },
                apply_result={"applied": True, "validation_ok": True},
            )

            stats = memory.get_local_judge_stats()
            confidence = stats["per_intent"]["general_judge"]["confidence"]
            self.assertGreater(confidence, 0.0)

            memory.update_local_judge_stats(
                info={
                    "intent": "general_judge",
                    "local_provider": "ollama_local",
                    "api_provider": "openai_dev",
                    "local_attempted": True,
                    "local_valid_json": True,
                    "escalated": False,
                    "local_patch_index": 0,
                    "api_patch_index": None,
                    "selected_source": "local",
                },
                apply_result={"applied": True, "validation_ok": False},
            )

            stats = memory.get_local_judge_stats()
            confidence_after = stats["per_intent"]["general_judge"]["confidence"]
            self.assertLessEqual(confidence_after, confidence)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from core.general_routing import decide_general_route, parse_local_assessment
from core.memory import MemoryStore


class GeneralRoutingTests(unittest.TestCase):
    def test_invalid_local_json_triggers_escalation(self) -> None:
        assessment, error = parse_local_assessment("not json")
        self.assertIsNone(assessment)
        self.assertEqual(error, "invalid_json")

        decision = decide_general_route(
            task="What is the latest version of Python?",
            assessment=assessment,
            confidence_threshold=0.90,
            web_first_threshold=0.70,
            learned_confidence=0.0,
        )
        self.assertEqual(decision["route"], "web")

    def test_factual_below_threshold_triggers_web(self) -> None:
        assessment = {
            "answer": "Not sure",
            "confidence": 0.5,
            "needs_web": False,
            "needs_api": False,
            "uncertainty_reasons": [],
            "suggested_search_query": "latest python version",
        }
        decision = decide_general_route(
            task="What is the latest version of Python?",
            assessment=assessment,
            confidence_threshold=0.90,
            web_first_threshold=0.70,
            learned_confidence=1.0,
        )
        self.assertEqual(decision["route"], "web")

    def test_non_factual_below_threshold_triggers_api(self) -> None:
        assessment = {
            "answer": "Draft response",
            "confidence": 0.5,
            "needs_web": False,
            "needs_api": False,
            "uncertainty_reasons": [],
            "suggested_search_query": "",
        }
        decision = decide_general_route(
            task="Write a concise project plan for a migration.",
            assessment=assessment,
            confidence_threshold=0.90,
            web_first_threshold=0.70,
            learned_confidence=1.0,
        )
        self.assertEqual(decision["route"], "api")

    def test_general_routing_stats_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.local.json"
            seed_path = Path(tmp) / "state.json"
            memory = MemoryStore(state_path=str(state_path), seed_path=str(seed_path))

            memory.update_general_routing_stats(
                info={
                    "intent": "factual",
                    "local_json_valid": True,
                    "local_only_answer": False,
                    "web_escalated": True,
                    "api_escalated": False,
                    "invalid_json": False,
                }
            )

            memory_reloaded = MemoryStore(state_path=str(state_path), seed_path=str(seed_path))
            stats = memory_reloaded.get_general_routing_stats()
            overall = stats.get("overall", {})
            self.assertEqual(overall.get("total_general_prompts"), 1)
            self.assertEqual(overall.get("web_escalations"), 1)


if __name__ == "__main__":
    unittest.main()

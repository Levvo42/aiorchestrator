import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.agent import Agent
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
            "final_answer": "Not sure",
            "confidence": 0.5,
            "escalate_to": "none",
            "search_query": "latest python version",
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
            "final_answer": "Draft response",
            "confidence": 0.5,
            "escalate_to": "none",
            "search_query": "",
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

    def test_hello_uses_ollama_only(self) -> None:
        class StubProvider:
            def __init__(self, response: str) -> None:
                self.response = response
                self.calls = []

            def generate(self, prompt: str) -> str:
                self.calls.append(prompt)
                return self.response

        capabilities = {
            "providers": {
                "ollama_local": {"enabled": True},
                "openai_dev": {"enabled": True},
            },
            "routing_rules": {},
            "judge": {"task_intent_keywords": {}},
            "dev": {},
        }

        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(
                state_path=str(Path(tmp) / "state.local.json"),
                seed_path=str(Path(tmp) / "state.json"),
            )
            memory.set_setting("general_mode", "auto")

            agent = Agent(capabilities=capabilities, memory=memory)

            local_json = (
                '{"final_answer": "Hello!", "confidence": 1.0, "escalate_to": "none", '
                '"search_query": "", "uncertainty_reasons": [], "suggested_search_query": ""}'
            )
            local = StubProvider(local_json)
            api = StubProvider("API answer")

            agent.provider_map = {"ollama_local": local, "openai_dev": api}

            run = agent.run("Hello!")

            self.assertEqual(len(local.calls), 1)
            self.assertEqual(len(api.calls), 0)
            self.assertEqual(run["routing"]["route"], "local")
            self.assertEqual(run["execution"]["llm"][0]["provider"], "ollama_local")

    def test_invalid_json_escalates(self) -> None:
        class StubProvider:
            def __init__(self, response: str) -> None:
                self.response = response
                self.calls = []

            def generate(self, prompt: str) -> str:
                self.calls.append(prompt)
                return self.response

        capabilities = {
            "providers": {
                "ollama_local": {"enabled": True},
                "openai_dev": {"enabled": True},
            },
            "routing_rules": {},
            "judge": {"task_intent_keywords": {}},
            "dev": {},
        }

        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(
                state_path=str(Path(tmp) / "state.local.json"),
                seed_path=str(Path(tmp) / "state.json"),
            )
            memory.set_setting("general_mode", "auto")

            agent = Agent(capabilities=capabilities, memory=memory)
            local = StubProvider("not json")
            api = StubProvider("API answer")
            agent.provider_map = {"ollama_local": local, "openai_dev": api}

            with patch("core.agent.web_search", return_value=[{"title": "t", "snippet": "s", "url": "u"}]):
                with patch("core.agent.web_search_vertex_status", return_value=([], "empty_results", "empty_results")):
                    run = agent.run("When was the digital clock invented?")

            self.assertEqual(len(local.calls), 2)
            self.assertEqual(len(api.calls), 0)
            self.assertEqual(run["routing"]["route"], "local->web")

    def test_digital_clock_invention_uses_web(self) -> None:
        class StubProvider:
            def __init__(self, responses: list[str]) -> None:
                self.responses = responses
                self.calls = []

            def generate(self, prompt: str) -> str:
                self.calls.append(prompt)
                return self.responses.pop(0)

        capabilities = {
            "providers": {
                "ollama_local": {"enabled": True},
                "openai_dev": {"enabled": True},
            },
            "routing_rules": {},
            "judge": {"task_intent_keywords": {}},
            "dev": {},
        }

        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(
                state_path=str(Path(tmp) / "state.local.json"),
                seed_path=str(Path(tmp) / "state.json"),
            )
            memory.set_setting("general_mode", "auto")

            agent = Agent(capabilities=capabilities, memory=memory)
            local = StubProvider([
                '{"final_answer": "Not sure", "confidence": 0.2, "escalate_to": "none", '
                '"search_query": "digital clock invented", "uncertainty_reasons": [], '
                '"suggested_search_query": ""}',
                "The digital clock was invented in the 1950s. Source: http://example.com",
            ])
            api = StubProvider(["API answer"])
            agent.provider_map = {"ollama_local": local, "openai_dev": api}

            with patch("core.agent.web_search", return_value=[{"title": "t", "snippet": "s", "url": "u"}]):
                with patch("core.agent.web_search_vertex_status", return_value=([], "empty_results", "empty_results")):
                    run = agent.run("When was the digital clock invented?")

            self.assertEqual(run["routing"]["route"], "local->web")
            self.assertEqual(run["routing"]["search_tool"], "open_web")
            self.assertEqual(len(api.calls), 0)

    def test_weather_query_uses_web(self) -> None:
        class StubProvider:
            def __init__(self, responses: list[str]) -> None:
                self.responses = responses
                self.calls = []

            def generate(self, prompt: str) -> str:
                self.calls.append(prompt)
                return self.responses.pop(0)

        capabilities = {
            "providers": {
                "ollama_local": {"enabled": True},
                "openai_dev": {"enabled": True},
            },
            "routing_rules": {},
            "judge": {"task_intent_keywords": {}},
            "dev": {},
        }

        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(
                state_path=str(Path(tmp) / "state.local.json"),
                seed_path=str(Path(tmp) / "state.json"),
            )
            memory.set_setting("general_mode", "auto")

            agent = Agent(capabilities=capabilities, memory=memory)
            local = StubProvider([
                '{"final_answer": "Not sure", "confidence": 0.2, "escalate_to": "none", '
                '"search_query": "weather Duved Sweden", "uncertainty_reasons": [], '
                '"suggested_search_query": ""}',
                "Weather details with Source: http://example.com",
            ])
            api = StubProvider(["API answer"])
            agent.provider_map = {"ollama_local": local, "openai_dev": api}

            with patch("core.agent.web_search", return_value=[{"title": "t", "snippet": "s", "url": "u"}]):
                with patch("core.agent.web_search_vertex_status", return_value=([], "empty_results", "empty_results")):
                    run = agent.run("How's the weather in Duved, Sweden?")

            self.assertEqual(run["routing"]["route"], "local->web")
            self.assertEqual(len(api.calls), 0)

    def test_web_error_escalates_to_api(self) -> None:
        class StubProvider:
            def __init__(self, response: str) -> None:
                self.response = response
                self.calls = []

            def generate(self, prompt: str) -> str:
                self.calls.append(prompt)
                return self.response

        capabilities = {
            "providers": {
                "ollama_local": {"enabled": True},
                "openai_dev": {"enabled": True},
            },
            "routing_rules": {},
            "judge": {"task_intent_keywords": {}},
            "dev": {},
        }

        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(
                state_path=str(Path(tmp) / "state.local.json"),
                seed_path=str(Path(tmp) / "state.json"),
            )
            memory.set_setting("general_mode", "auto")

            agent = Agent(capabilities=capabilities, memory=memory)
            local = StubProvider(
                '{"final_answer": "Not sure", "confidence": 0.2, "escalate_to": "web", '
                '"search_query": "weather Duved Sweden", "uncertainty_reasons": [], '
                '"suggested_search_query": ""}'
            )
            api = StubProvider("API answer")
            agent.provider_map = {"ollama_local": local, "openai_dev": api}

            with patch("core.agent.web_search", return_value=[]):
                with patch("core.agent.web_search_vertex_status", return_value=([], "error", "boom")):
                    run = agent.run("How's the weather in Duved, Sweden?")

            self.assertEqual(run["routing"]["route"], "local->web->api")
            self.assertGreater(len(api.calls), 0)

    def test_authoritative_query_uses_vertex_first(self) -> None:
        class StubProvider:
            def __init__(self, responses: list[str]) -> None:
                self.responses = responses
                self.calls = []

            def generate(self, prompt: str) -> str:
                self.calls.append(prompt)
                return self.responses.pop(0)

        capabilities = {
            "providers": {
                "ollama_local": {"enabled": True},
                "openai_dev": {"enabled": True},
            },
            "routing_rules": {},
            "judge": {"task_intent_keywords": {}},
            "dev": {},
        }

        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(
                state_path=str(Path(tmp) / "state.local.json"),
                seed_path=str(Path(tmp) / "state.json"),
            )
            memory.set_setting("general_mode", "auto")

            agent = Agent(capabilities=capabilities, memory=memory)
            local = StubProvider([
                '{"final_answer": "Not sure", "confidence": 0.2, "escalate_to": "none", '
                '"search_query": "ibuprofen side effects", "uncertainty_reasons": [], '
                '"suggested_search_query": ""}',
                "Side effects with Source: http://example.com",
            ])
            api = StubProvider(["API answer"])
            agent.provider_map = {"ollama_local": local, "openai_dev": api}

            with patch("core.agent.web_search_vertex_status", return_value=([{"title": "t", "snippet": "s", "url": "u"}], "ok", "")) as vertex_mock:
                with patch("core.agent.web_search", return_value=[]) as open_mock:
                    run = agent.run("What are the side effects of ibuprofen?")

            self.assertEqual(run["routing"]["search_tool"], "vertex_search")
            self.assertEqual(len(api.calls), 0)
            self.assertEqual(vertex_mock.call_count, 1)
            self.assertEqual(open_mock.call_count, 0)


if __name__ == "__main__":
    unittest.main()

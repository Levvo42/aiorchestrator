import json
import os
import unittest
from unittest.mock import patch

from tools.web_search_google import web_search_google


class _FakeResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class WebSearchGoogleTests(unittest.TestCase):
    def test_query_encoded_and_normalized(self) -> None:
        payload = {
            "items": [
                {"title": "Title A", "snippet": "Snippet A", "link": "https://a.example"},
                {"title": "Title B", "snippet": "Snippet B", "link": "https://b.example"},
            ]
        }
        fake = _FakeResponse(json.dumps(payload))

        with patch.dict(os.environ, {"GOOGLE_CSE_API_KEY": "k", "GOOGLE_CSE_CX": "cx"}):
            with patch("tools.web_search_google.urllib.request.urlopen", return_value=fake) as mocked:
                results = web_search_google("digital clock invented", num_results=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "Title A")
        self.assertEqual(results[0]["snippet"], "Snippet A")
        self.assertEqual(results[0]["url"], "https://a.example")

        request = mocked.call_args[0][0]
        self.assertIn("q=digital+clock+invented", request.full_url)
        self.assertIn("num=2", request.full_url)


if __name__ == "__main__":
    unittest.main()

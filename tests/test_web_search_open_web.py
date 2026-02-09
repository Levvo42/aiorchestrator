import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from tools.web_search_open_web import web_search_open_web_status


class _FakeResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class WebSearchOpenWebTests(unittest.TestCase):
    def test_brave_normalization(self) -> None:
        payload = {
            "web": {
                "results": [
                    {
                        "title": "Brave Result",
                        "url": "https://brave.com/search",
                        "description": "Search the web.",
                    }
                ]
            }
        }
        fake_resp = _FakeResponse(json.dumps(payload))

        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "token"}):
            with patch("tools.web_search_open_web.urllib.request.urlopen", return_value=fake_resp) as mocked:
                results, status, _ = web_search_open_web_status("brave search", num_results=1)

        self.assertEqual(status, "ok")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Brave Result")
        self.assertEqual(results[0]["snippet"], "Search the web.")
        self.assertEqual(results[0]["url"], "https://brave.com/search")
        self.assertEqual(results[0]["source_domain"], "brave.com")

        request = mocked.call_args[0][0]
        self.assertIn("q=brave+search", request.full_url)

    def test_brave_auth_failed(self) -> None:
        err = urllib.error.HTTPError(
            "https://api.search.brave.com/res/v1/web/search",
            401,
            "Unauthorized",
            None,
            io.BytesIO(b"{}"),
        )

        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "bad"}):
            with patch("tools.web_search_open_web.urllib.request.urlopen", side_effect=err):
                results, status, detail = web_search_open_web_status("brave search", num_results=1)

        self.assertEqual(results, [])
        self.assertEqual(status, "auth_failed")
        self.assertEqual(detail, "http_401")

    def test_brave_rate_limited(self) -> None:
        err = urllib.error.HTTPError(
            "https://api.search.brave.com/res/v1/web/search",
            429,
            "Too Many Requests",
            None,
            io.BytesIO(b"{}"),
        )

        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "token"}):
            with patch("tools.web_search_open_web.urllib.request.urlopen", side_effect=err):
                results, status, detail = web_search_open_web_status("brave search", num_results=1)

        self.assertEqual(results, [])
        self.assertEqual(status, "rate_limited")
        self.assertEqual(detail, "http_429")


if __name__ == "__main__":
    unittest.main()

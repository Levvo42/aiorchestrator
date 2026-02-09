import json
import os
import types
import unittest
from unittest.mock import patch

from tools.web_search_vertex import web_search_vertex_status


class _FakeResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class WebSearchVertexTests(unittest.TestCase):
    def test_vertex_normalization(self) -> None:
        payload = {
            "results": [
                {
                    "document": {
                        "derivedStructData": {
                            "title": "Title A",
                            "snippet": "Snippet A",
                            "link": "https://a.example/path",
                        }
                    }
                }
            ]
        }
        fake_resp = _FakeResponse(json.dumps(payload))

        class FakeCreds:
            def __init__(self) -> None:
                self.token = "token"

            def refresh(self, _req) -> None:
                return None

        fake_auth = types.SimpleNamespace(
            default=lambda scopes=None: (FakeCreds(), None)
        )
        fake_requests = types.SimpleNamespace(Request=object)
        fake_transport = types.SimpleNamespace(requests=fake_requests)
        fake_google = types.SimpleNamespace(auth=fake_auth)

        with patch.dict(
            os.environ,
            {
                "VERTEX_PROJECT_ID": "proj",
                "VERTEX_LOCATION": "global",
                "VERTEX_COLLECTION": "default_collection",
                "VERTEX_ENGINE_ID": "engine",
                "VERTEX_SERVING_CONFIG": "default_search",
                "VERTEX_DATA_STORE_ID": "ds",
            },
        ):
            with patch.dict("sys.modules", {
                "google": fake_google,
                "google.auth": fake_auth,
                "google.auth.transport": fake_transport,
                "google.auth.transport.requests": fake_requests,
            }):
                with patch("tools.web_search_vertex.urllib.request.urlopen", return_value=fake_resp) as mocked:
                    results, status, _ = web_search_vertex_status("ibuprofen side effects", num_results=1)

        self.assertEqual(status, "ok")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Title A")
        self.assertEqual(results[0]["snippet"], "Snippet A")
        self.assertEqual(results[0]["url"], "https://a.example/path")
        self.assertEqual(results[0]["source_domain"], "a.example")

        request = mocked.call_args[0][0]
        self.assertIn("projects/proj/locations/global/collections/default_collection/engines/engine", request.full_url)


if __name__ == "__main__":
    unittest.main()

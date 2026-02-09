AI-Orchestrator

Runtime
- Local LLM (Ollama) handles general tasks with web escalation as needed.

Search Providers
- Vertex AI Search (Discovery Engine) is used for trusted retrieval only (no generative features).
- Open web search uses Brave Search API for general queries when trusted search is not required.

Authentication
- Vertex AI Search uses Application Default Credentials via `GOOGLE_APPLICATION_CREDENTIALS`.
- Brave Search uses `BRAVE_SEARCH_API_KEY` (optional override `BRAVE_SEARCH_ENDPOINT`).
- Do not store API keys in code; configure `.env` or shell env vars.

See `docs/config.md` for required environment variables.

AI-Orchestrator

Search Providers
- Google Custom Search (CSE) has been removed.
- Vertex AI Search (Discovery Engine) is used for trusted retrieval only (no generative features).
- Open web search is used for general queries when trusted search is not required.

Authentication
- Vertex AI Search uses Application Default Credentials via `GOOGLE_APPLICATION_CREDENTIALS`.
- Do not store API keys in code; configure `.env` or shell env vars.

See `docs/config.md` for required environment variables.

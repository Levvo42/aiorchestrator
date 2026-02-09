Configuration

Vertex AI Search (Discovery Engine)
- `GOOGLE_APPLICATION_CREDENTIALS` (service account path)
- `VERTEX_PROJECT_ID`
- `VERTEX_LOCATION` (use `global`)
- `VERTEX_COLLECTION` (default: `default_collection`)
- `VERTEX_ENGINE_ID`
- `VERTEX_SERVING_CONFIG` (default: `default_search`)
- `VERTEX_DATA_STORE_ID`

Notes
- Vertex AI Search is retrieval-only; no generative APIs are used.
- If Vertex AI Search fails or returns empty results, the router may fall back to open web search or API LLMs depending on the query type.

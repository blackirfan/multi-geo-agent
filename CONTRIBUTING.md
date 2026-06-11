# Contributing to GeoReasoner

Thank you for your interest in contributing. This document covers how to set up a development environment, run the test suite, and submit changes.

---

## Development Setup

```bash
git clone https://github.com/your-org/geo-multi-agent.git
cd geo-multi-agent

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

On macOS, the WeasyPrint PDF backend requires Pango:

```bash
brew install pango
```

On Debian/Ubuntu:

```bash
sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0 libcairo2
```

---

## Running Tests

```bash
pytest tests/ -v --cov=georeasoner --cov-report=term-missing
```

The full suite (90 tests) runs without LM Studio. Every agent node has a hard fallback that calls geospatial tools directly when `get_llm()` fails.

To test against a live LM Studio instance, start it at `http://localhost:1234/v1` before running pytest. The tests detect the connection and switch from fallback mode automatically.

---

## Linting

```bash
ruff check georeasoner tests scripts
```

Fix auto-correctable issues with:

```bash
ruff check --fix georeasoner tests scripts
```

All lint errors must be zero before submitting a PR.

---

## Code Style

- Python 3.12+, full type hints everywhere
- Pydantic models for all API request/response bodies
- No in-memory GeoDataFrames crossing agent boundaries — pass file paths via `GeoReasonerState`
- `ensure_*()` functions in `data_utils.py` for any new data dependency (real data priority, synthetic fallback)
- Every new agent node must implement the three-layer pattern: LLM tool-calling → hard fallback → trace + state update

---

## Adding a New Agent Node

1. Create `georeasoner/agents/my_agent.py`
2. Implement `my_agent_node(state: GeoReasonerState) -> dict` following the three-layer pattern
3. Register new state fields in `georeasoner/state.py`
4. Wire the node into `georeasoner/graph.py` via `builder.add_node` and `builder.add_edge`
5. Add tests in `tests/test_phaseN.py` that patch `get_llm` to raise `ConnectionError` (verifies fallback)

---

## Adding a New Tool

1. Add the function to the appropriate module in `georeasoner/tools/`
2. Decorate with `@tool` (LangChain) or expose as a plain callable used by the fallback
3. Ensure it returns a JSON-serialisable string (not a GeoDataFrame)
4. Write a unit test that calls it directly with synthetic data

---

## Pull Request Checklist

- [ ] `pytest` passes with zero failures
- [ ] `ruff check` reports zero errors
- [ ] New public functions have type hints
- [ ] New agent nodes include a fallback path
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

---

## Project Structure Quick Reference

```
georeasoner/
├── agents/         # Five LangGraph node functions
├── tools/          # Geospatial tools (vector, raster, hydrology)
├── api/            # FastAPI app + route handlers
├── templates/      # Jinja2 HTML report template
├── static/         # Leaflet fallback page
├── data_utils.py   # ensure_*() data availability helpers
├── graph.py        # LangGraph StateGraph assembly
├── state.py        # GeoReasonerState TypedDict
├── report_writer.py# PDF + HTML report generation
└── db.py           # DuckDB spatial extension init
```

---

## Commit Messages

Use the imperative mood in the subject line, keep it under 72 characters. Reference the phase or module:

```
feat(hydrology): add NDWI-based water body detection
fix(api): handle missing report gracefully in GET /reports/{run_id}
test(phase4): add report writer unit tests
```

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

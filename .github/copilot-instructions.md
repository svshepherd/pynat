# Copilot Instructions

## Project context
This repository contains iNaturalist helper workflows with optional REST fallback behavior.

## Environment and dependencies
- Use `uv` for environment and dependency operations.
- Prefer commands: `uv sync`, `uv run python ...`, and `uv run pytest`.
- Keep API keys out of committed files.

## Code style expectations
- Preserve fallback behavior when `pyinaturalist` is unavailable.
- Keep user-facing dataframe columns stable unless a migration note is added.
- Avoid introducing mandatory network calls in utility functions unless documented.

## Testing expectations
- Mock API calls in tests; do not depend on live API responses.
- Ensure both pyinaturalist and REST-fallback paths are covered for changed logic.

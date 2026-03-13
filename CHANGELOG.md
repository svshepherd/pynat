# Changelog

## Unreleased

### Added
- `load_api_key()` helper: runtime API key loading from env, keyring, or fallback dill file.
- Unit tests with pytest and simple mocks: `tests/test_helpers.py`.

### Changed
- Removed import-time unpickling of the API key (no side-effects on import).
- `get_mine()`: runtime date defaults, robust `observed_on` parsing, non-blocking image handling.
- `coming_soon()` and `get_park_data()`: numeric coercion of counts, chunked API requests, and safe normalization to avoid division-by-zero.
- Replaced bare `except:` with specific exception handling and added structured logging.

### Fixed
- Various crash scenarios when API returns empty/malformed results.

### Notes
- Test suite is run via `uv` dev extras (for example: `uv run --extra dev pytest -q`).
- Current local baseline: 27 passing tests.

# Copilot Instructions for climate-canvas

Trust these instructions. Only search the codebase if information here is missing or found to be incorrect — this repo is small and most things are documented below.

## Repository Summary

**climate-canvas** is a small Python library + CLI for plotting climate impact assessment "response surfaces" and other climate change scenario visualizations (2D contour plots from CSV data of x/y/z values). It is a dependency of the sibling repo `hydropattern`.

- **Size**: tiny (~6 source files, 2 test files).
- **Type**: Python package + Typer-based CLI (`climate-canvas` console script).
- **Language/runtime**: Python **3.12+** (strict `requires-python = ">=3.12"`).
- **Key libs**: `typer` (CLI), `pandas`/`numpy` (data), `matplotlib` (plotting).
- **Env/build tool**: **uv** (migrated from Poetry — `poetry.lock` is gone, use `uv.lock`/`pyproject.toml`).
- **No linter or type checker is configured** for this project (no ruff/mypy dependency, no config for either). Do not add lint/type-check steps unless explicitly asked.

## Build, Test, and Run — Always Use These Exact Steps

Always run commands from the repo root. Always run `uv sync` before anything else after cloning or after any `pyproject.toml`/dependency change.

```bash
# 1. Bootstrap environment (creates/updates .venv, installs deps incl. test group)
uv sync --group test

# 2. Run the full test suite
uv run pytest -q

# 3. Run CLI (entry point defined in [project.scripts])
uv run climate-canvas --help
uv run climate-canvas response --help
uv run climate-canvas response examples\scenario_data.csv
uv run climate-canvas response examples\scenario_data.csv --interp

# 4. Build distributable artifacts (sdist + wheel), if needed
uv build
```

All four steps above were validated to work in this environment with `uv 0.9.26` and Python 3.12. `uv sync --group test` takes well under a minute normally.

### Known pre-existing test failure (not caused by your changes)
`tests/test_format_data.py::TestFormatData::test_format_data_files` fails on a clean checkout with:
```
AssertionError: False is not true
```
It asserts an output CSV exists at a path relative to CWD that isn't produced under normal `pytest` invocation from repo root — this is a pre-existing bug in the test, unrelated to environment setup. **Do not treat this single failure as caused by your change**; if you touch `data_utilities.py`/`format_data_files`, verify by comparing against this known baseline (1 failed, rest passed) rather than assuming your change broke it.

### No lint/type-check step
There is no `ruff`, `mypy`, `flake8`, or `pylint` dependency declared and no config sections for them in `pyproject.toml`. CI does not run any lint step. Do not invent one.

## CI / Validation Pipeline

`.github/workflows/ci.yml` (GitHub Actions, `ubuntu-latest`) runs on every push to `main`/`master` and on every PR:
1. Checkout
2. `astral-sh/setup-uv@v6` (installs uv)
3. `actions/setup-python@v5` pinned to `"3.12"`
4. `uv sync --group test`
5. `uv run pytest`

Replicate this exact sequence locally before considering a change done. There is currently no separate build/lint/type-check job — passing `uv run pytest` (minus the one known pre-existing failure above) is the bar for CI.

## Project Layout

```
climate-canvas/
├── climate_canvas/            # Package source
│   ├── __init__.py
│   ├── __main__.py            # `python -m climate_canvas` entry (delegates to cli.app)
│   ├── cli.py                 # Typer app; `response` command is the only CLI command
│   ├── data_utilities.py      # CSV/data reshaping helpers (format_data_files, etc.)
│   └── plots_utilities.py     # matplotlib plotting logic (plot_response_surface)
├── tests/
│   ├── test_data_utilities.py
│   └── test_format_data.py    # contains the known pre-existing failing test (see above)
├── examples/                  # Example CSVs used in README and manual testing
│   └── scenario_data.csv, complex_surface.csv, multiple_response_surfaces/...
├── img/                       # README screenshots/plots
├── pyproject.toml             # uv/hatchling project + dependency-groups config
├── uv.lock                    # uv lockfile (authoritative dependency versions)
├── .github/workflows/ci.yml   # CI pipeline (see above)
└── README.md                  # Install (uv) + usage instructions, kept in sync with CLI
```

### Entry points
- **Console script**: `climate-canvas` (see `[project.scripts]` in `pyproject.toml`) → `climate_canvas.cli:app`.
- **Module entry**: `climate_canvas/__main__.py` → same Typer `app`, run via `python -m climate_canvas`.
- **Main CLI logic**: `climate_canvas/cli.py` — single `response` command; reads a CSV into a pandas DataFrame, splits into x/y/z numpy arrays, and calls `plot_response_surface` in `plots_utilities.py`.

### Dependency notes (not obvious from file layout)
- `typer` is pinned `>=0.12.3,<0.13.0` **and** `click` is explicitly pinned `>=8.1.7,<8.2.0`. This is required: newer `click` (8.2+/8.4+) breaks `typer` 0.12.x's `--help` rendering with `TypeError: Parameter.make_metavar() missing 1 required positional argument: 'ctx'`. If you ever touch dependency versions, keep this click pin (or upgrade typer in lockstep) — do not remove it.
- `dependency-groups.test` includes `ipykernel` (used for notebook-based manual exploration, not part of test execution itself).
- There is no `dev` dependency group in this repo (unlike the sibling `hydropattern` repo which has one for mypy/ruff).

### Configuration files
- `pyproject.toml`: single source of truth for project metadata, dependencies, build backend (`hatchling`), and console script. No `[tool.ruff]`/`[tool.mypy]` sections exist.
- `uv.lock`: exact resolved dependency versions; regenerate via `uv sync` if `pyproject.toml` changes (never hand-edit).
- No `.flake8`, `.pylintrc`, `mypy.ini`, or `ruff.toml` exist — don't look for them.

## Guidance for Changes
- When adding/modifying CLI options, update both `climate_canvas/cli.py` and the corresponding examples/usage text in `README.md` (README commands are kept runnable/verified, e.g. `uv run climate-canvas response examples\scenario_data.csv --interp`).
- When adding dependencies, add them to `pyproject.toml` `[project.dependencies]` or the appropriate `[dependency-groups]` entry, then run `uv sync --group test` to update `uv.lock` — always commit the updated `uv.lock` alongside `pyproject.toml` changes.
- Tests live under `tests/` and mirror module names in `climate_canvas/` (e.g. `data_utilities.py` ↔ `test_data_utilities.py`). Add new tests in the matching file/pattern.

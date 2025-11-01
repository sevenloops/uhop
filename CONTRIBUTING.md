# Contributing to UHOP

Thanks for your interest in contributing! This guide explains how to set up your environment, run tests, and open pull requests that get merged quickly.

## Quick start

- Use the helper script to set up, test, and run:
  - `./contributing.sh setup` — create venv and install dependencies (editable + dev)
  - `./contributing.sh test` — run unit tests (GPU paths are skipped/forced to baseline by default)
  - `./contributing.sh frontend:build` — build docs/demo site
  - `./contributing.sh api` — start the demo API (<http://127.0.0.1:5824>)
  - `./contributing.sh bridge` — start the local bridge (<http://127.0.0.1:5823>)

If the script isn’t executable yet:

```bash
chmod +x contributing.sh
```

Windows notes:

- Use Git Bash (or WSL) to run the script. If you prefer PowerShell/CMD, you can do the steps manually (see below).
- The script auto-detects Python as `python3` or `python` and activates venv on Windows (`.venv/Scripts/activate`).

## Development environment

- Python 3.10+ (project uses 3.11 in CI)
- Node.js 18+ for the docs/demo frontend (or Bun 1.x)

Create a virtualenv and install dev dependencies:

```bash
./contributing.sh setup
```

Manual setup (if not using the script):

```bash
# Linux/macOS
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Running tests

- Python tests (fast):

```bash
./contributing.sh test
```

This sets `UHOP_FORCE_BASELINE=1` so GPU/cuda/opencl code paths don’t make tests flaky in local dev.

- Frontend build (ensures TypeScript/ESLint basics pass during build):

```bash
./contributing.sh frontend:build
```

The script uses Bun if available, otherwise falls back to npm.

## Pull request guidelines

Install pre-commit hooks locally (runs ruff, black, isort, flake8, prettier where present):

```bash
pip install pre-commit
pre-commit install
# or via helper
./contributing.sh hooks
```

- Keep PRs focused and small where possible.
- Write clear titles (prefer Conventional Commit style, e.g., `feat: ...`, `fix: ...`, `docs: ...`, `test: ...`).
- Include a short summary of what changed and why.
- Add or update tests for user-visible changes.
- Update docs (README/CONTRIBUTING) when changing UX or commands.
- Ensure CI is green before requesting review.

## Code style

- Python: PEP8 with modern tooling. We recommend: ruff, black, isort (installed with the `dev` extra).
- TypeScript/React: keep components small and typed; prefer composition over complexity.

## CI & reviews

- All PRs run (GitHub Actions):
  - Python lint (ruff) and unit tests (pytest)
  - Frontend lint (eslint) and build (Vite)
- PR template prompts for summary, checks, and validation steps.
- PR titles should follow Conventional Commits; you can edit the title if needed.

## Security & secrets

- Never commit secrets or tokens.
- Use environment variables or local `.env` files that are .gitignored.

## Release process (lightweight)

- We’ll squash-merge with a sensible Conventional Commit title for changelog clarity.
- Tags and releases are done by maintainers.

Thanks again! If you’re unsure whether a change is desired, open an issue first to discuss.

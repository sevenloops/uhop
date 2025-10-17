# Contributing to UHOP

Thanks for your interest in contributing! This guide explains how to set up your environment, run tests, and open pull requests that get merged quickly.

## Quick start

- Use the helper script to set up, test, and run:
  - `./contributing.sh setup` — create venv and install dependencies (editable + dev)
  - `./contributing.sh test` — run unit tests
  - `./contributing.sh frontend:build` — build docs/demo site
  - `./contributing.sh api` — start the demo API (<http://127.0.0.1:5824>)
  - `./contributing.sh bridge` — start the local bridge (<http://127.0.0.1:5823>)

If the script isn’t executable yet:

```bash
chmod +x contributing.sh
```

## Development environment

- Python 3.10+ (project uses 3.11 in CI)
- Node.js 18+ for the docs/demo frontend

Create a virtualenv and install dev dependencies:

```bash
./contributing.sh setup
```

## Running tests

- Python tests:

```bash
./contributing.sh test
```

- Frontend build (ensures TypeScript/ESLint basics pass during build):

```bash
./contributing.sh frontend:build
```

## Pull request guidelines
 
 Consider installing pre-commit hooks locally:

```bash
pip install pre-commit
pre-commit install
 This will run black, isort, flake8, and prettier on changed files before each commit.
This will run black, isort, flake8, and prettier on changed files before each commit.


- Keep PRs focused and small where possible.
- Write clear titles (prefer Conventional Commit style, e.g., `feat: ...`, `fix: ...`, `docs: ...`, `test: ...`).
- Include a short summary of what changed and why.
- Add or update tests for user-visible changes.
- Update docs (README/CONTRIBUTING) when changing UX or commands.
- Ensure CI is green before requesting review.

## Code style

- Python: aim for PEP8. If you use `black` and `isort`, great; we don’t strictly fail CI on style yet.
- TypeScript/React: keep components small and typed; prefer composition over complexity.

## CI & reviews

- All PRs run:
  - Python tests (`pytest`)
  - Frontend build (Vite)
- PR template will prompt for summary, changes, and checklist.
- CODEOWNERS requires `@sevenloops` to review changes to core paths.
- PR titles are checked for Conventional Commit prefixes; fix-up is easy if you forget — just edit the title.

## Security & secrets

- Never commit secrets or tokens.
- Use environment variables or local `.env` files that are .gitignored.

## Release process (lightweight)

- We’ll squash-merge with a sensible Conventional Commit title for changelog clarity.
- Tags and releases are done by maintainers.

Thanks again! If you’re unsure whether a change is desired, open an issue first to discuss.

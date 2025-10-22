# Improve CI/CD workflow, docs, and release hygiene

Goal: Make UHOP friendlier for contributions and promotions by polishing the end-to-end contributor experience and tightening CI/CD. This is a great Good First Issue for open-source specialists who enjoy refining docs and workflows.

## Scope

- CI workflows
  - Add/verify GitHub Actions on PRs: Python tests (matrix: Windows/macOS/Linux, Python 3.10–3.12), frontend build (Node 20)
  - Cache dependencies (pip, node) for faster runs; set timeouts and clear logs
  - Optional job: markdown lint and link check for docs/README
  - Optional job: check Conventional Commit titles (or lenient warning)
- Docs & badges
  - Add status badges (CI, coverage if available, website/pages) to README
  - Add “How to run tests locally” and “How to debug CI” snippets in CONTRIBUTING.md
  - Ensure README Getting Started commands match the CLI
- Templates & metadata
  - Review/refresh `.github/pull_request_template.md` for clarity
  - Add issue templates (bug report, feature request) if missing
  - Consider adding `CODEOWNERS` for core paths and well-chosen labels (good first issue, help wanted)
- Releases & pages
  - (Optional) Draft a lightweight release checklist (CHANGELOG or GitHub Releases notes)
  - Confirm GitHub Pages deploy for frontend; ensure README links point to working URLs

## Acceptance criteria

- Green CI for default branch with the matrix above
- PRs show clear statuses and reasonable runtimes (< 10 min typical)
- README has the requested badges and consistent commands
- CONTRIBUTING.md includes a short CI/PR troubleshooting section
- PR template prompts for summary, screenshots/benchmarks (when relevant), and checklist

## References

- Existing scripts: `contributing.sh`, `README.md`, `CONTRIBUTING.md`
- CLI entry points: `uhop/cli.py`
- Frontend: `frontend/` (Vite + Tailwind)

If you’re interested, comment to claim and outline your plan before opening a PR. Thanks!

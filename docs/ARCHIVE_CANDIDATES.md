# Docs archive candidates

These documents are useful for context but not critical to day-to-day contributors. Consider moving them to `docs/archive/` or pruning after extracting any evergreen content.

- `docs/PROJECT_CONTEXT_FOR_COPILOT.md`
  - AI assistant context dump; not needed for users/contributors.
- `docs/RUN_REPORT.md`
  - One-off run/perf notes; replace with KPI snapshot tooling and GitHub issue comments.
- `docs/OSS_HELP_WANTED_ISSUE.md`
  - Template material; could be kept in `.github/ISSUE_TEMPLATE/` instead.
- `docs/DEPLOY.md`
  - Minimal Web API deploy instructions duplicated in README; consolidate in one place.
- `docs/EDGE_DEPLOYMENT.md`
  - Keep if we intend to support "edge lite" path; otherwise, fold into a single deployment guide.

Keep and continue to update:

- `docs/ENV_VARS.md`
- `docs/AUTOTUNE.md`
- `docs/CLBLAST_BUILD.md`
- `docs/AGENT_QUICKSTART.md`
- `docs/IR_MVP.md`
- `docs/PROTOCOL_V0_1.md`

Action:

- Create `docs/archive/` and move the candidates above there, updating links as needed in a follow-up PR.

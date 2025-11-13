# Production Release Checklist

This checklist is designed for maintainers preparing an enterprise-facing UHOP release. Duplicate the table into your release notes or tracking issue and mark each item as completed before publishing binaries or tagged builds.

## How to Use

1. Make a copy of the checklist in your release planning document (GitHub issue, Notion page, etc.).
2. Assign owners and due dates per item.
3. Update the **Status** and **Notes** columns as you work through the release.
4. Only cut the release when every row is marked ✅.

## Checklist Template

| Area                                                                          | Owner | Status | Notes |
| ----------------------------------------------------------------------------- | ----- | ------ | ----- |
| Security sandbox enforced (`UHOP_ENFORCE_SANDBOX=1` default, fallback tested) |       |        |       |
| Deterministic build hashes verified (`uhop cache verify --all`)               |       |        |       |
| Automation metrics reviewed (Issue 20 report)                                 |       |        |       |
| Observability hooks emitting required signals (logs + KPI snapshot)           |       |        |       |
| Release checklist CI job green                                                |       |        |       |
| Docs updated (`README`, `docs/ROADMAP.md`, relevant feature docs)             |       |        |       |
| `docs/PRODUCTION_VISION.md` changes summarized in release notes               |       |        |       |
| Customer/partner comms drafted and scheduled                                  |       |        |       |
| Rollback plan validated (fast path to disable new kernels)                    |       |        |       |

## Sample (Filled)

| Area                                                                          | Owner  | Status | Notes                                                                                               |
| ----------------------------------------------------------------------------- | ------ | ------ | --------------------------------------------------------------------------------------------------- |
| Security sandbox enforced (`UHOP_ENFORCE_SANDBOX=1` default, fallback tested) | @alice | ✅     | Sandbox defaulted in config commit 1a2b3c; fallback test link.                                      |
| Deterministic build hashes verified (`uhop cache verify --all`)               | @bob   | ✅     | Hash diff tool output attached to release issue.                                                    |
| Automation metrics reviewed (Issue 20 report)                                 | @carol | ✅     | `automation_report --since 14d` shows 89% promotion success.                                        |
| Observability hooks emitting required signals (logs + KPI snapshot)           | @dave  | ✅     | Grafana dashboard `uhop-prod` updated, alerts firing.                                               |
| Release checklist CI job green                                                | @eve   | ✅     | Workflow run [GitHub Actions 123456789](https://github.com/sevenloops/uhop/actions/runs/123456789). |
| Docs updated (`README`, `docs/ROADMAP.md`, relevant feature docs)             | @frank | ✅     | Docs PR #456 merged.                                                                                |
| `docs/PRODUCTION_VISION.md` changes summarized in release notes               | @grace | ✅     | Section added under "Vision updates" in release notes draft.                                        |
| Customer/partner comms drafted and scheduled                                  | @heidi | 🟡     | Draft shared with marketing, pending approval.                                                      |
| Rollback plan validated (fast path to disable new kernels)                    | @ivan  | ✅     | Runbook entry updated; tested `UHOP_FORCE_BASELINE=1` toggle.                                       |

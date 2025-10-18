# UHop MVP (developer-integrated)

[![Deploy Frontend to GitHub Pages](https://github.com/sevenloops/uhop/actions/workflows/deploy-frontend-pages.yml/badge.svg)](https://github.com/sevenloops/uhop/actions/workflows/deploy-frontend-pages.yml)

Live site: <https://uhop.dev>

This repository is an MVP for UHOP — a runtime that:

- Detects hardware (CUDA vs CPU),
- Runs hand-written CUDA kernels (if PyCUDA available),
- Generates CUDA kernels via OpenAI (if configured),
- Benchmarks and caches the best implementation,
- Exposes a decorator `@hop.optimize("matmul")` for easy integration.

## Quickstart

1. Clone project.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   # or at minimum:
   pip install numpy openai
   # Optional for CUDA:
   pip install pycuda
   ```

## CLI

After installing this package in your environment, you get a `uhop` command:

- `uhop info` — print detected hardware and backend availability (Torch, Triton, OpenCL).
- `uhop info --json` — machine-readable JSON hardware info (includes backend availability).
- `uhop info --ocl-device 0` — override OpenCL GPU device selection by index (across all platforms).
- `uhop demo --size 192` — run a quick Naive Python vs UHOP-optimized matmul benchmark and show which wins.
- `uhop demo --iters 5 --ocl-device 0` — adjust iterations and explicitly choose an OpenCL GPU.

Environment override: set `UHOP_OPENCL_DEVICE_INDEX=<idx>` to select a default OpenCL device for the session.

## extra demos

- MatMul (UHOP vs naive):
  - `uhop demo --size 192`
  - Expect UHOP to win decisively vs naive Python; increase size for more stress.
- Fused Conv2D+ReLU (OpenCL fused kernel):
  - `python -m uhop.cli demo-conv2d-relu --h 128 --w 128 --c-in 3 --c-out 32 --k 3 --stride 1 --padding 1`
  - Use `--ocl-device` to select a GPU, and try larger shapes (e.g. `--h 224 --w 224 --c-in 16 --c-out 32`) for stronger GPU gains.

See `docs/RUN_REPORT.md` for a summary of errors encountered and solutions applied during development, plus sample benchmark outputs.

## Syncing GitHub Issues from `issues/` (optional)

## Online demo API (optional)

To let people try the demo directly on the website (without running a local bridge), you can run a tiny HTTP API that exposes safe endpoints:

Endpoints:

- GET /health
- GET /info — same JSON as `uhop info --json`
- POST /demo/matmul — runs a small, bounded matmul demo and returns timings

Run locally:

```bash
uhop web-api --host 0.0.0.0 --port 5824
# or
python -m uhop.web_api --host 0.0.0.0 --port 5824
```

Docker:

```bash
docker build -t uhop-demo-api -f api.Dockerfile .
docker run --rm -p 5824:5824 uhop-demo-api
```

Point the docs/demo site to this API by setting `VITE_UHOP_API_BASE`, e.g.:

```bash
VITE_UHOP_API_BASE="https://demo-api.uhop.dev" npm run build
```


This repo includes a GitHub Actions workflow that automatically creates and updates GitHub Issues based on Markdown files under the `issues/` folder.

## Deploying the public demo

Want YC reviewers to try the demo online without local setup? Deploy the backend to Railway/Render/Fly and the frontend to Vercel or GitHub Pages. See `docs/DEPLOY.md` for step-by-step instructions.

What it does:

- On every push that changes files in `issues/**` (or when manually triggered), the workflow scans `issues/` for `.md` files.
- For each file, it creates or updates a GitHub Issue with a stable footer marker: `Source: issues/<path/to/file.md>`.
- It adds the label `synced-from-folder` to identify issues managed by CI.
- If a source file is deleted, the corresponding Issue is automatically closed with a comment.

File format:

- Title will be derived from front matter (`title:`), the first `# Heading` in the Markdown, or the filename.
- Optional front matter is supported between `---` lines at the top. Supported keys:
  - `title: <string>`
  - `labels: [label-a, label-b]` or `labels: label-a, label-b`
  - `assignees: [user1, user2]` or `assignees: user1, user2`

Example (`issues/01-example.md`):

```markdown
---
title: Improve GPU kernel selection
labels: [optimization, mvp]
assignees: user1, user2
---

# Improve GPU kernel selection

Describe the change, acceptance criteria, and references here.
```

How to run it:

- Push changes to files under `issues/` to trigger the workflow automatically.
- Or run it manually from the Actions tab: "Sync GitHub Issues from files" → Run workflow.

Notes:

- The workflow adds a footer to the issue body with the source path and a warning not to edit below the separator. Edit the Markdown file instead.
- The workflow merges any existing labels with those from front matter and always ensures the `synced-from-folder` label is present.

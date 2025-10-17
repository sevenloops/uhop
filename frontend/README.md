# UHOP Kernel Forge – Docs & Demo Site

This is the documentation and demo website for the UHOP project.

• Production repo: <https://github.com/sevenloops/uhop>

The site showcases:

- What UHOP is, how it works, and architecture
- Detailed docs for CLI and Python APIs
- MPS (Apple), CUDA (Torch), OpenCL, and optional Triton notes
- A simulated interactive demo with copyable local commands

## Tech Stack

- Vite + React + TypeScript
- Tailwind CSS + shadcn/ui
- Recharts (demo visualization)

## Prerequisites

- Node.js 18+ recommended

## Run locally

```bash
npm install
npm run dev
```

Open the local URL printed by Vite (usually <http://localhost:5173>).

Optionally set an online API base so the Demo page can run without a local bridge:

```bash
VITE_UHOP_API_BASE=https://demo-api.uhop.dev npm run dev
```

## Build

```bash
npm run build
npm run preview
```

For deployment, you can bake the API base into the build:

```bash
VITE_UHOP_API_BASE=https://demo-api.uhop.dev npm run build
```

## Content sources

This site is derived from the core UHOP repository; commands and APIs reference the Python package available on PyPI (uhop). See the main README and docs within the monorepo for deeper technical details.

## Contributing

PRs welcome. Keep content aligned with the actual CLI and APIs:

- CLI: `uhop info`, `uhop demo`, `uhop demo-conv2d-relu`, `uhop cache *`, `uhop ai-generate`, `uhop ai-generate-fused`
- Python: `optimize`, `UHopOptimizer`, `detect_hardware`, `UhopCache`, `validation.validate_callable`

## License

MIT (same as UHOP)

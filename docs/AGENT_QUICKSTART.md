# UHOP Agent — Quickstart

Run the UHOP demo on your own hardware by pairing a lightweight local agent with the hosted portal.

## What it does

- Detects your local hardware (CUDA/OpenCL/CPU/MPS) and exposes it to the demo portal
- Executes benchmarks and kernel generation locally (your data never leaves your machine)
- Optional: uses your own `OPENAI_API_KEY` for AI kernel generation

## Requirements

- Python 3.10+ on Windows/macOS/Linux
- Optional GPU drivers/toolkits if you want GPU acceleration (CUDA, vendor OpenCL, etc.)

## Install

```bash
pip install uhop  # or: pip install -e .  (from the repo)
```

## Start the agent

Replace `api.yourdomain.com` with the backend URL provided by the portal (for development use `localhost`).

```bash
# Production (TLS):
#uhop-agent --server wss://api.yourdomain.com/agent --token YOUR_AGENT_TOKEN

# Local development:
uhop-agent --server ws://127.0.0.1:8787/agent
```

- `--token` (optional but recommended): the server can require a shared pairing token (`AGENT_TOKEN`).
- The agent will auto-reconnect by default; add `--no-reconnect` to run once.
- Add `--debug` for verbose WebSocket tracing when diagnosing issues.

## Use the portal

1. Open the demo portal page
2. The Status panel should show `Agent: Connected`
3. Click “Connect Devices”, then run benchmarks — they will execute on your machine

## AI kernel generation with your key (optional)

Set your API key before launching the agent:

```bash
# macOS/Linux
export OPENAI_API_KEY=sk-...
uhop-agent --server wss://api.yourdomain.com/agent --token YOUR_AGENT_TOKEN

# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-..."
uhop-agent --server wss://api.yourdomain.com/agent --token YOUR_AGENT_TOKEN
```

## Troubleshooting

- WebSocket blocked at work? Ensure outbound TCP 443 is allowed and proxies support WebSockets (WSS). Try a different network.
- Timeouts on `localhost`: try `--server ws://127.0.0.1:8787/agent` (avoids IPv6/localhost quirks)
- GPU not detected: install vendor drivers/toolkits and retry; check `uhop info --json` locally
- Server says token mismatch: confirm `AGENT_TOKEN` on server matches your `--token`
- Firewall prompts: allow the agent to make outbound connections

## Privacy & Security

- The agent only communicates with your chosen backend URL via (WSS/WS)
- Computation runs on your machine when the agent is connected
- Use tokens (`AGENT_TOKEN` / `--token`) to restrict which agents can pair

## Useful commands

```bash
uhop info --json           # local hardware snapshot
uhop-agent --debug ...     # verbose agent logs
```

## Advanced: IR CLI and lowering

If you want to experiment locally with the intermediate representation (IR) path and OpenCL lowering used by the agent, see "UHOP IR and Scheduling" and the IR MVP doc:

- docs/UHOP_IR_AND_SCHEDULING.md
- docs/IR_MVP.md (includes an IR CLI with lower/build/validate/bench commands)

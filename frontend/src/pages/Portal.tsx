import { useEffect, useMemo, useState } from "react";

type Device = {
  id: string;
  type: "CPU" | "GPU" | "Edge";
  name: string;
  status: string;
};
const envAny = import.meta as unknown as { env?: Record<string, string> };
const API_BASE: string =
  envAny?.env?.VITE_BACKEND_URL || "http://localhost:8787";

export default function Portal() {
  const [devices, setDevices] = useState<Device[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [result, setResult] = useState<{
    device: string;
    time: number;
    timings?: { uhop: number; naive: number; uhop_won: boolean };
  } | null>(null);
  const [code, setCode] = useState<string>("");
  const [wsOk, setWsOk] = useState<boolean>(false);
  const [health, setHealth] = useState<string>("unknown");
  const [agentConnected, setAgentConnected] = useState<boolean>(false);
  const [matrixSize, setMatrixSize] = useState<number>(128);
  const [iterations, setIterations] = useState<number>(2);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [isConnecting, setIsConnecting] = useState<boolean>(false);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [showHelp, setShowHelp] = useState<boolean>(false);

  useEffect(() => {
    let closed = false;
    let retryTimer: number | undefined;
    function connect() {
      try {
        const url = API_BASE.replace(/^http/, "ws");
        const ws = new WebSocket(url);
        ws.onopen = () => setWsOk(true);
        ws.onmessage = (ev) => {
          try {
            const { line } = JSON.parse(String(ev.data));
            if (!closed && line) setLogs((prev) => [...prev, line].slice(-500));
          } catch (_e) {
            /* ignore */
          }
        };
        ws.onclose = () => {
          setWsOk(false);
          if (!closed) {
            retryTimer = window.setTimeout(connect, 1500) as unknown as number;
          }
        };
        return ws;
      } catch (_e) {
        setWsOk(false);
        retryTimer = window.setTimeout(connect, 1500) as unknown as number;
        return null;
      }
    }
    const ws = connect();
    return () => {
      closed = true;
      try {
        if (ws) ws.close();
      } catch (_e) {
        /* noop */
      }
      if (retryTimer) window.clearTimeout(retryTimer);
    };
  }, []);

  useEffect(() => {
    let stop = false;
    async function poll() {
      try {
        const r = await fetch(`${API_BASE}/agent-status`);
        const j = await r.json();
        if (!stop) setAgentConnected(!!j.connected);
      } catch (_e) {
        if (!stop) setAgentConnected(false);
      }
      if (!stop) setTimeout(poll, 3000);
    }
    poll();
    return () => {
      stop = true;
    };
  }, []);

  async function connectDevices() {
    if (isConnecting) return;
    setIsConnecting(true);
    try {
      const r = await fetch(`${API_BASE}/connect`, { method: "POST" });
      const j = await r.json();
      if (j.devices) {
        setDevices(j.devices);
      } else if (j.error) {
        setLogs((prev) => [...prev, `[Error] ${j.error}`].slice(-500));
      }
    } catch (error) {
      setLogs((prev) =>
        [...prev, `[Error] Failed to connect: ${error}`].slice(-500),
      );
    } finally {
      setIsConnecting(false);
    }
  }

  async function runDemo() {
    if (isRunning) return;
    setIsRunning(true);
    try {
      const r = await fetch(`${API_BASE}/run-demo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ size: matrixSize, iters: iterations }),
      });
      const j = await r.json();
      if (j && j.result === "success") {
        setResult({
          device: j.device,
          time: parseFloat(j.time),
          timings: j.timings,
        });
      } else if (j.error) {
        setLogs((prev) => [...prev, `[Error] ${j.error}`].slice(-500));
      }
    } catch (error) {
      setLogs((prev) => [...prev, `[Error] Demo failed: ${error}`].slice(-500));
    } finally {
      setIsRunning(false);
    }
  }

  async function genKernel() {
    if (isGenerating) return;
    setIsGenerating(true);
    try {
      const r = await fetch(`${API_BASE}/generate-kernel`, { method: "POST" });
      const j = await r.json();
      setCode(j.code || "");
    } catch (error) {
      setLogs((prev) =>
        [...prev, `[Error] Kernel generation failed: ${error}`].slice(-500),
      );
    } finally {
      setIsGenerating(false);
    }
  }

  async function checkHealth() {
    try {
      const r = await fetch(`${API_BASE}/health`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      setHealth(j.ok ? "ok" : "bad");
    } catch (_e) {
      setHealth("down");
    }
  }

  const deviceDots = useMemo(() => {
    const colors: Record<string, string> = {
      CPU: "#6ee7b7",
      GPU: "#93c5fd",
      Edge: "#fca5a5",
    };
    return devices.map((d, i) => (
      <g key={d.id} transform={`translate(${40 + i * 60},40)`}>
        <circle
          r={14}
          fill={colors[d.type] || "#fff"}
          opacity={d.status === "connected" ? 1 : 0.3}
        />
        <text x={0} y={28} textAnchor="middle" fill="#94a3b8" fontSize="10">
          {d.type}
        </text>
      </g>
    ));
  }, [devices]);

  useEffect(() => {
    document.title = "U-HOP Demo Portal";
  }, []);

  return (
    <div className="min-h-screen bg-[#0b1020] text-[#e5e7eb] font-mono">
      <div className="max-w-6xl mx-auto p-6">
        <header className="flex items-center justify-between mb-6">
          <h1 className="text-2xl md:text-3xl font-bold tracking-wide">
            U-HOP Demo Portal
          </h1>
          <div className="text-xs text-[#94a3b8]">
            Sevenloops — U-HOP·{" "}
            <a
              className="underline"
              href="https://github.com/sevenloops/uhop"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>
          </div>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Help / Instructions */}
          <section className="md:col-span-3">
            <button
              className="text-left w-full px-3 py-2 rounded bg-[#0f172a] border border-[#374151] hover:bg-[#111827]"
              onClick={() => setShowHelp((s) => !s)}
            >
              <div className="flex items-center justify-between">
                <span className="font-semibold">How to use this demo</span>
                <span className="text-[#94a3b8] text-sm">
                  {showHelp ? "Hide" : "Show"}
                </span>
              </div>
            </button>
            {showHelp && (
              <div className="mt-2 p-3 rounded border border-[#374151] bg-[#0b1220] text-sm space-y-3">
                <div>
                  This portal can run benchmarks on either the hosted server or
                  your own machine via a lightweight "UHOP Agent".
                </div>
                <div className="space-y-1">
                  <div className="font-semibold">
                    Option A — Hosted (no install)
                  </div>
                  <ul className="list-disc pl-5 space-y-1 text-[#9ca3af]">
                    <li>
                      Click{" "}
                      <span className="text-[#e5e7eb]">Connect Devices</span> to
                      detect the server hardware.
                    </li>
                    <li>Adjust Matrix Size / Iterations if desired.</li>
                    <li>
                      Click{" "}
                      <span className="text-[#e5e7eb]">Run Matmul Test</span> to
                      benchmark on the server.
                    </li>
                    <li>
                      Click{" "}
                      <span className="text-[#e5e7eb]">Generate AI Kernel</span>{" "}
                      to view a generated kernel. If the server has no AI key, a
                      fallback kernel is shown.
                    </li>
                  </ul>
                </div>
                <div className="space-y-1">
                  <div className="font-semibold">
                    Option B — Local Agent (run on your PC)
                  </div>
                  <ul className="list-disc pl-5 space-y-1 text-[#9ca3af]">
                    <li>
                      Requirements: Python 3.10+, and optional GPU
                      drivers/toolkits if you want GPU acceleration.
                    </li>
                    <li>
                      Install UHOP:{" "}
                      <code className="bg-[#111827] px-1 py-0.5 rounded">
                        pip install uhop
                      </code>{" "}
                      (or from source if developing).
                    </li>
                    <li>Start the agent while this page is open:</li>
                  </ul>
                  <div className="grid md:grid-cols-3 gap-3 text-xs">
                    <div className="border border-[#374151] rounded p-2">
                      <div className="text-[#93c5fd] mb-1">
                        Windows (PowerShell)
                      </div>
                      <pre className="whitespace-pre-wrap">
                        <code>
                          uhop-agent --server ws://your-server-host:8787/agent
                        </code>
                      </pre>
                    </div>
                    <div className="border border-[#374151] rounded p-2">
                      <div className="text-[#93c5fd] mb-1">macOS</div>
                      <pre className="whitespace-pre-wrap">
                        <code>
                          uhop-agent --server ws://your-server-host:8787/agent
                        </code>
                      </pre>
                    </div>
                    <div className="border border-[#374151] rounded p-2">
                      <div className="text-[#93c5fd] mb-1">Linux</div>
                      <pre className="whitespace-pre-wrap">
                        <code>
                          uhop-agent --server ws://your-server-host:8787/agent
                        </code>
                      </pre>
                    </div>
                  </div>
                  <ul className="list-disc pl-5 space-y-1 text-[#9ca3af] mt-2">
                    <li>
                      When the agent connects, the status below will show{" "}
                      <span className="text-[#6ee7b7]">Agent: Connected</span>.
                      Then use the same buttons as above — runs will execute on
                      your machine.
                    </li>
                    <li>
                      Optional security: set a token on the server (
                      <code className="bg-[#111827] px-1 py-0.5 rounded">
                        AGENT_TOKEN
                      </code>
                      ) and pass it to the agent (
                      <code className="bg-[#111827] px-1 py-0.5 rounded">
                        --token
                      </code>
                      ).
                    </li>
                    <li>
                      For AI kernel generation with your own API key, set{" "}
                      <code className="bg-[#111827] px-1 py-0.5 rounded">
                        OPENAI_API_KEY
                      </code>{" "}
                      in your shell before starting the agent.
                    </li>
                  </ul>
                </div>
                <div className="space-y-1">
                  <div className="font-semibold">Notes</div>
                  <ul className="list-disc pl-5 space-y-1 text-[#9ca3af]">
                    <li>
                      We limit matrix size and iterations in the demo for
                      safety.
                    </li>
                    <li>
                      UHOP chooses backends based on detected hardware
                      (CUDA/OpenCL/CPU/MPS when available).
                    </li>
                    <li>
                      If something fails, check the System Log Console for
                      details.
                    </li>
                  </ul>
                </div>
              </div>
            )}
          </section>
          {/* Controls */}
          <section className="md:col-span-1 space-y-4">
            <h3 className="text-lg font-semibold text-[#e5e7eb] mb-3">
              Controls
            </h3>

            {/* Configuration */}
            <div className="space-y-3 p-3 rounded border border-[#374151] bg-[#0f172a]">
              <div className="text-sm text-[#94a3b8] mb-2">Configuration</div>
              <div>
                <label className="block text-xs text-[#9ca3af] mb-1">
                  Matrix Size
                </label>
                <input
                  type="number"
                  value={matrixSize}
                  onChange={(e) =>
                    setMatrixSize(parseInt(e.target.value) || 128)
                  }
                  min="16"
                  max="512"
                  step="16"
                  placeholder="128"
                  title="Matrix size for benchmark"
                  className="w-full px-2 py-1 text-xs bg-[#111827] border border-[#374151] rounded text-[#e5e7eb]"
                />
              </div>
              <div>
                <label className="block text-xs text-[#9ca3af] mb-1">
                  Iterations
                </label>
                <input
                  type="number"
                  value={iterations}
                  onChange={(e) => setIterations(parseInt(e.target.value) || 2)}
                  min="1"
                  max="10"
                  placeholder="2"
                  title="Number of benchmark iterations"
                  className="w-full px-2 py-1 text-xs bg-[#111827] border border-[#374151] rounded text-[#e5e7eb]"
                />
              </div>
            </div>

            {/* Action Buttons */}
            <div className="space-y-2">
              <button
                className="w-full py-2 rounded bg-[#111827] hover:bg-[#1f2937] border border-[#374151] disabled:opacity-50 disabled:cursor-not-allowed"
                onClick={connectDevices}
                disabled={isConnecting}
              >
                {isConnecting ? "Connecting..." : "Connect Devices"}
              </button>
              <button
                className="w-full py-2 rounded bg-[#111827] hover:bg-[#1f2937] border border-[#374151] disabled:opacity-50 disabled:cursor-not-allowed"
                onClick={runDemo}
                disabled={isRunning || devices.length === 0}
              >
                {isRunning
                  ? "Running..."
                  : `Run Matmul Test (${matrixSize}x${matrixSize})`}
              </button>
              <button
                className="w-full py-2 rounded bg-[#111827] hover:bg-[#1f2937] border border-[#374151] disabled:opacity-50 disabled:cursor-not-allowed"
                onClick={genKernel}
                disabled={isGenerating}
              >
                {isGenerating ? "Generating..." : "Generate AI Kernel"}
              </button>
              <button
                className="w-full py-2 rounded bg-[#0b1324] hover:bg-[#111c34] border border-[#374151]"
                onClick={checkHealth}
              >
                Health Check
              </button>
            </div>

            {/* Status */}
            <div className="text-xs text-[#9ca3af] space-y-1">
              <div>Backend: {API_BASE}</div>
              <div>WS: {wsOk ? "Connected" : "Disconnected"}</div>
              <div>Health: {health}</div>
              <div>Agent: {agentConnected ? "Connected" : "Not connected"}</div>
              <div>Devices: {devices.length} connected</div>
            </div>

            {/* Device Visualization */}
            <svg width="100%" height="100" viewBox="0 0 320 100">
              <rect
                x="0"
                y="0"
                width="100%"
                height="100%"
                fill="transparent"
                stroke="#334155"
              />
              {deviceDots}
            </svg>
          </section>

          {/* System Log Console */}
          <section className="md:col-span-2">
            <div className="text-sm mb-2">System Log Console</div>
            <div className="h-64 overflow-auto rounded border border-[#374151] bg-[#0f172a] p-3 text-xs whitespace-pre-wrap">
              {logs.length === 0 ? (
                <div className="text-[#64748b]">Awaiting events...</div>
              ) : (
                logs.map((l, i) => <div key={i}>{l}</div>)
              )}
            </div>
          </section>

          {/* Result panel */}
          <section className="md:col-span-1">
            <div className="text-sm mb-2">Benchmark Results</div>
            <div className="rounded border border-[#374151] bg-[#0f172a] p-3 text-xs space-y-2">
              {result ? (
                <>
                  <div>
                    Device:{" "}
                    <span className="text-[#93c5fd]">{result.device}</span>
                  </div>
                  <div>
                    UHOP Time:{" "}
                    <span className="text-[#6ee7b7]">{result.time} ms</span>
                  </div>
                  {result.timings && (
                    <>
                      <div className="border-t border-[#374151] pt-2 mt-2">
                        <div className="text-[#9ca3af] text-xs mb-1">
                          Detailed Timings:
                        </div>
                        <div>
                          UHOP:{" "}
                          <span className="text-[#6ee7b7]">
                            {(result.timings.uhop * 1000).toFixed(2)} ms
                          </span>
                        </div>
                        <div>
                          Naive:{" "}
                          <span className="text-[#fca5a5]">
                            {(result.timings.naive * 1000).toFixed(2)} ms
                          </span>
                        </div>
                        <div>
                          Speedup:{" "}
                          <span className="text-[#fbbf24]">
                            {(
                              result.timings.naive / result.timings.uhop
                            ).toFixed(1)}
                            x
                          </span>
                        </div>
                        <div
                          className={`mt-1 ${result.timings.uhop_won ? "text-[#6ee7b7]" : "text-[#fca5a5]"}`}
                        >
                          {result.timings.uhop_won
                            ? "✅ UHOP Wins!"
                            : "⚠️ Baseline was faster"}
                        </div>
                      </div>
                    </>
                  )}
                </>
              ) : (
                <div className="text-[#64748b]">
                  Run a benchmark to see results
                </div>
              )}
            </div>
          </section>

          {/* Generated Code View */}
          <section className="md:col-span-2">
            <div className="text-sm mb-2">AI Generated Kernel Code</div>
            <pre className="rounded border border-[#374151] bg-[#0b1220] p-3 text-xs overflow-auto max-h-64">
              <code className="text-[#e5e7eb]">
                {code ||
                  '// Click "Generate AI Kernel" to view an AI-generated CUDA/OpenCL kernel\n// Real UHOP AI will generate optimized kernels based on your hardware'}
              </code>
            </pre>
          </section>
        </div>
      </div>
    </div>
  );
}

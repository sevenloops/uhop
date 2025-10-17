import { useEffect, useMemo, useRef, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import CodeBlock from "@/components/CodeBlock";
import { Play, Cpu, Zap } from "lucide-react";

const Demo = () => {
  // Interactive CLI-style demo (no charts). Users run commands locally and paste outputs here.
  const [selectedOperation, setSelectedOperation] = useState("matmul");
  const [hardwareJson, setHardwareJson] = useState("");
  const [demoOut, setDemoOut] = useState("");
  const [fusedOut, setFusedOut] = useState("");
  const [aiGenOut, setAiGenOut] = useState("");
  const [bridgeUrl, setBridgeUrl] = useState("http://127.0.0.1:5823");
  const [bridgeStatus, setBridgeStatus] = useState<"unknown"|"checking"|"ok"|"down">("unknown");
  const [bridgeMsg, setBridgeMsg] = useState<string>("");

  async function pingBridge(urlOverride?: string) {
    const url = urlOverride ?? bridgeUrl;
    setBridgeStatus("checking");
    setBridgeMsg("");
    try {
      const r = await fetch(`${url}/health`);
      if (r.ok) {
        setBridgeStatus("ok");
        if (urlOverride) setBridgeUrl(urlOverride);
      } else {
        setBridgeStatus("down");
        setBridgeMsg(`Health check failed with status ${r.status}`);
      }
    } catch (e) {
      setBridgeStatus("down");
      setBridgeMsg("Could not reach bridge. Is it running?");
    }
  }

  // Auto-detect on first load: try 127.0.0.1 then localhost
  const autoTriedRef = useRef(false);
  useEffect(() => {
    if (autoTriedRef.current) return;
    autoTriedRef.current = true;
    let cancelled = false;
    (async () => {
      // prefer 127.0.0.1
      try {
        const r1 = await fetch("http://127.0.0.1:5823/health");
        if (!cancelled && r1.ok) {
          setBridgeUrl("http://127.0.0.1:5823");
          setBridgeStatus("ok");
          return;
        }
      } catch (e) {
        // ignored
      }
      // fallback to localhost (may resolve to ::1)
      try {
        const r2 = await fetch("http://localhost:5823/health");
        if (!cancelled && r2.ok) {
          setBridgeUrl("http://localhost:5823");
          setBridgeStatus("ok");
          return;
        }
      } catch (e) {
        // ignored
      }
      if (!cancelled) setBridgeStatus("down");
    })();
    return () => { cancelled = true; };
  }, []);

  async function runCmd(cmd: string): Promise<{code:number,stdout:string,stderr:string}|null> {
    try {
      const r = await fetch(`${bridgeUrl}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cmd })
      });
      if (!r.ok) {
        setBridgeStatus("down");
        setBridgeMsg(`Bridge responded with status ${r.status}`);
        return null;
      }
      return await r.json();
    } catch (e) {
      setBridgeStatus("down");
      setBridgeMsg("Failed to contact bridge at " + bridgeUrl);
      return null;
    }
  }

  const operations = [
    { id: "matmul", name: "Matrix Multiplication", desc: "Dense matrix-matrix product" },
    { id: "conv2d", name: "2D Convolution", desc: "Convolutional neural network operation" },
    { id: "relu", name: "ReLU Activation", desc: "Rectified Linear Unit activation" },
  ];

  // Parsers for pasted outputs
  const parsedHardware = useMemo(() => {
    try {
      const obj = JSON.parse(hardwareJson);
      return {
        vendor: obj.vendor,
        kind: obj.kind,
        name: obj.name,
        torch_available: obj.torch_available,
        torch_mps_available: obj.torch_mps_available,
        torch_preferred_device: obj.torch_preferred_device,
        triton_available: obj.triton_available,
        opencl_available: obj.opencl_available,
        raw: obj,
      };
    } catch (_) {
      return null;
    }
  }, [hardwareJson]);

  function parseDemoTimings(text: string) {
    // Look for lines like: "UHOP (optimized over naive): 0.012345 s" and "Naive Python baseline     : 1.234567 s"
    const uh = /UHOP \(optimized over naive\):\s*([0-9.]+) s/.exec(text);
    const nv = /Naive Python baseline\s*:\s*([0-9.]+) s/.exec(text);
    return {
      uhop: uh ? parseFloat(uh[1]) : undefined,
      naive: nv ? parseFloat(nv[1]) : undefined,
    };
  }

  const demoTimings = useMemo(() => parseDemoTimings(demoOut), [demoOut]);

  return (
    <div className="min-h-screen pt-16">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="max-w-6xl mx-auto space-y-8">
          <div className="text-center space-y-4">
            <h1 className="text-4xl sm:text-5xl font-bold gradient-text">
              Interactive Demo
            </h1>
            <p className="text-xl text-muted-foreground">
              Watch UHop discover and optimize your hardware in real time
            </p>
          </div>

          {/* Operation Selection */}
          <Card className="p-6 bg-card/50 backdrop-blur-sm">
            <h2 className="text-2xl font-semibold mb-4">Select Operation</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {operations.map((op) => (
                <button
                  key={op.id}
                  onClick={() => setSelectedOperation(op.id)}
                  className={`p-4 rounded-lg border-2 transition-all text-left ${
                    selectedOperation === op.id
                      ? "border-primary bg-primary/10"
                      : "border-border hover:border-primary/50"
                  }`}
                >
                  <h3 className="font-semibold mb-1">{op.name}</h3>
                  <p className="text-sm text-muted-foreground">{op.desc}</p>
                </button>
              ))}
            </div>
          </Card>

          {/* Hardware detection (CLI-guided) */}
          <Card className="p-6 bg-card/50 backdrop-blur-sm">
            <div className="flex items-center gap-3 mb-4">
              <Cpu className="h-5 w-5 text-primary" />
              <h2 className="text-2xl font-semibold">Detect Hardware (CLI)</h2>
            </div>
            <p className="text-sm text-muted-foreground mb-3">
              Option A) Click "Run via Bridge" (requires local bridge). Option B) Run the command
              yourself and paste the output.
            </p>
            <CodeBlock code={`uhop info --json`} language="bash" />
            <div className="mt-2 flex gap-2 items-center flex-wrap">
              <Button variant="outline" size="sm" onClick={()=>pingBridge()}>Check Bridge</Button>
              <Button variant="ghost" size="sm" onClick={()=>pingBridge("http://127.0.0.1:5823")}>Try 127.0.0.1</Button>
              <Button variant="ghost" size="sm" onClick={()=>pingBridge("http://localhost:5823")}>Try localhost</Button>
              <span className={`text-xs ${bridgeStatus==='ok'?'text-green-600':bridgeStatus==='down'?'text-red-600':bridgeStatus==='checking'?'text-amber-600':'text-muted-foreground'}`}>
                Bridge: {bridgeStatus}
              </span>
              <input
                value={bridgeUrl}
                onChange={(e)=>setBridgeUrl(e.target.value)}
                className="ml-auto px-2 py-1 rounded border text-xs bg-muted/30 w-64"
                placeholder="http://127.0.0.1:5823"
              />
              <Button
                size="sm"
                onClick={async()=>{
                  const resp = await runCmd('uhop info --json');
                  if (resp && resp.code === 0) setHardwareJson(resp.stdout);
                }}
              >Run via Bridge</Button>
            </div>
            {(bridgeStatus==='down' || bridgeStatus==='unknown') && (
              <div className="mt-2 text-xs text-muted-foreground space-y-2">
                {bridgeMsg && <div className="text-red-600">{bridgeMsg}</div>}
                <div>
                  To enable "Run via Bridge", start the local bridge in a terminal:
                </div>
                <CodeBlock code={`uhop web-bridge --port 5823`} language="bash" />
                <div className="text-xs">or</div>
                <CodeBlock code={`python -m uhop.web_bridge --port 5823`} language="bash" />
              </div>
            )}
            <textarea
              value={hardwareJson}
              onChange={(e) => setHardwareJson(e.target.value)}
              placeholder="Paste JSON output from 'uhop info --json'..."
              className="mt-3 w-full h-40 p-3 rounded-md border bg-muted/30 font-mono text-sm"
            />
            {parsedHardware && (
              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
                <Card className="p-3"><div className="text-sm">Vendor: <strong>{String(parsedHardware.vendor)}</strong></div></Card>
                <Card className="p-3"><div className="text-sm">Kind: <strong>{String(parsedHardware.kind)}</strong></div></Card>
                <Card className="p-3"><div className="text-sm">Name: <strong>{String(parsedHardware.name)}</strong></div></Card>
                <Card className="p-3"><div className="text-sm">Torch available: <strong>{String(parsedHardware.torch_available)}</strong></div></Card>
                <Card className="p-3"><div className="text-sm">Torch MPS: <strong>{String(parsedHardware.torch_mps_available)}</strong></div></Card>
                <Card className="p-3"><div className="text-sm">Torch preferred: <strong>{String(parsedHardware.torch_preferred_device)}</strong></div></Card>
                <Card className="p-3"><div className="text-sm">OpenCL: <strong>{String(parsedHardware.opencl_available)}</strong></div></Card>
                <Card className="p-3"><div className="text-sm">Triton: <strong>{String(parsedHardware.triton_available)}</strong></div></Card>
              </div>
            )}
          </Card>

          {/* Matmul demo (CLI-guided) */}
          <Card className="p-6 bg-card/50 backdrop-blur-sm">
            <div className="flex items-center gap-3 mb-4">
              <Play className="h-5 w-5 text-primary" />
              <h2 className="text-2xl font-semibold">Run Demo (CLI)</h2>
            </div>
            <p className="text-sm text-muted-foreground mb-3">Option A) Run via local bridge. Option B) run locally and paste output.</p>
            <CodeBlock code={`uhop demo --size 256 --iters 3`} language="bash" />
            <div className="mt-2 flex gap-2">
              <Button
                size="sm"
                onClick={async()=>{
                  const resp = await runCmd('uhop demo --size 256 --iters 3');
                  if (resp && resp.code === 0) setDemoOut(resp.stdout);
                }}
              >Run via Bridge</Button>
            </div>
            <textarea
              value={demoOut}
              onChange={(e) => setDemoOut(e.target.value)}
              placeholder="Paste output from 'uhop demo'..."
              className="mt-3 w-full h-40 p-3 rounded-md border bg-muted/30 font-mono text-sm"
            />
            {(demoTimings.uhop !== undefined || demoTimings.naive !== undefined) && (
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
                {demoTimings.uhop !== undefined && (
                  <Card className="p-3"><div className="text-sm">UHOP time: <strong>{demoTimings.uhop}s</strong></div></Card>
                )}
                {demoTimings.naive !== undefined && (
                  <Card className="p-3"><div className="text-sm">Naive baseline: <strong>{demoTimings.naive}s</strong></div></Card>
                )}
              </div>
            )}
          </Card>

          {/* AI generation (CLI-guided) */}
          <Card className="p-6 bg-card/50 backdrop-blur-sm">
            <div className="flex items-center gap-3 mb-4">
              <Zap className="h-5 w-5 text-secondary" />
              <h2 className="text-2xl font-semibold">AI Kernel Generation (CLI)</h2>
            </div>
            <p className="text-sm text-muted-foreground mb-3">Generate candidates and run a quick smoke test on your hardware:</p>
            <div className="space-y-3">
              <CodeBlock code={`uhop ai-generate --operation matmul --target opencl`} language="bash" />
              <CodeBlock code={`uhop ai-generate-fused --target opencl`} language="bash" />
            </div>
            <div className="mt-2 flex gap-2">
              <Button
                size="sm"
                onClick={async()=>{
                  const resp = await runCmd('uhop ai-generate --operation matmul --target opencl');
                  if (resp && resp.code === 0) setAiGenOut(resp.stdout + (resp.stderr?`\nSTDERR:\n${resp.stderr}`:''));
                }}
              >Generate (matmul)</Button>
              <Button
                size="sm"
                variant="outline"
                onClick={async()=>{
                  const resp = await runCmd('uhop ai-generate-fused --target opencl');
                  if (resp && resp.code === 0) setAiGenOut(resp.stdout + (resp.stderr?`\nSTDERR:\n${resp.stderr}`:''));
                }}
              >Generate (fused)</Button>
            </div>
            <textarea
              value={aiGenOut}
              onChange={(e) => setAiGenOut(e.target.value)}
              placeholder="Paste output from 'uhop ai-generate*'..."
              className="mt-3 w-full h-40 p-3 rounded-md border bg-muted/30 font-mono text-sm"
            />
            {!!aiGenOut && (
              <p className="text-xs text-muted-foreground mt-3">
                Output captured. UHOP only adopts AI-generated kernels if they pass validation; use --strict-validate for tighter tolerances.
              </p>
            )}
          </Card>

          {/* Try it locally */}
          <Card className="p-6 bg-card/50 backdrop-blur-sm">
            <h2 className="text-2xl font-semibold mb-4">Try It Locally</h2>
            <p className="text-muted-foreground mb-4">Run these commands on your machine to test UHOP with your hardware:</p>
            <div className="space-y-3">
              <CodeBlock code={`uhop info --json`} language="bash" />
              <CodeBlock code={`uhop demo --size 256 --iters 3`} language="bash" />
              <CodeBlock code={`uhop demo-conv2d-relu --c-in 3 --c-out 16 --h 64 --w 64 --k 3`} language="bash" />
              <CodeBlock code={`uhop cache list`} language="bash" />
            </div>
            <p className="text-xs text-muted-foreground mt-3">Note: The in-browser demo is a simulation for visualization; run commands locally for real performance and results.</p>
          </Card>

          {/* Removed chart visualization and static code preview to focus on real CLI-driven flows */}
        </div>
      </div>
    </div>
  );
};

export default Demo;

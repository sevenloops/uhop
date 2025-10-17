import { useState } from "react";
import { Link } from "react-router-dom";
import { ChevronRight, Book, Rocket, Code, Cog, FileText, Zap } from "lucide-react";
import { Card } from "@/components/ui/card";
import CodeBlock from "@/components/CodeBlock";

const Docs = () => {
  const [activeSection, setActiveSection] = useState("introduction");

  const sections = [
    { id: "introduction", title: "Introduction", icon: Book },
    { id: "getting-started", title: "Getting Started", icon: Rocket },
    { id: "core-concepts", title: "Core Concepts", icon: Code },
    { id: "cli-usage", title: "CLI Usage", icon: Cog },
    { id: "developer-guide", title: "Developer Guide", icon: FileText },
    { id: "api-reference", title: "API Reference", icon: Zap },
  ];

  return (
    <div className="min-h-screen pt-16">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="flex gap-8">
          {/* Sidebar */}
          <aside className="hidden lg:block w-64 shrink-0">
            <nav className="sticky top-24 space-y-1">
              {sections.map((section) => (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`w-full flex items-center gap-3 px-4 py-2 rounded-lg text-left transition-colors ${
                    activeSection === section.id
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  }`}
                >
                  <section.icon className="h-4 w-4" />
                  <span className="text-sm font-medium">{section.title}</span>
                </button>
              ))}
            </nav>
          </aside>

          {/* Main Content */}
          <main className="flex-1 max-w-4xl">
            {activeSection === "introduction" && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h1 className="text-4xl font-bold mb-4">Introduction to UHOP</h1>
                  <p className="text-lg text-muted-foreground">
                    Universal Hardware Optimization Protocol
                  </p>
                </div>

                <Card className="p-6 bg-card/50 border-primary/20">
                  <h2 className="text-2xl font-semibold mb-4">What is UHOP?</h2>
                  <p className="text-muted-foreground leading-relaxed mb-4">
                    UHOP is an AI-driven hardware optimization runtime that automatically detects your computing hardware,
                    prioritizes accelerators (CUDA, Apple MPS, OpenCL) when available, generates or selects optimized
                    implementations for core ops, validates correctness, benchmarks their performance, and caches the
                    fastest choices for reuse.
                  </p>
                  <p className="text-muted-foreground leading-relaxed">
                    It provides a universal abstraction layer across different hardware backends (Torch CUDA/MPS/CPU,
                    OpenCL, optional Triton), enabling developers to write once and run optimally everywhere.
                  </p>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Why UHop Matters</h2>
                  <ul className="space-y-3 text-muted-foreground">
                    <li className="flex items-start gap-3">
                      <ChevronRight className="h-5 w-5 text-primary mt-0.5 shrink-0" />
                      <span><strong className="text-foreground">Hardware Agnostic:</strong> Write code once, optimize for any device automatically</span>
                    </li>
                    <li className="flex items-start gap-3">
                      <ChevronRight className="h-5 w-5 text-primary mt-0.5 shrink-0" />
                      <span><strong className="text-foreground">AI-Powered:</strong> Leverages machine learning to generate and validate optimal kernels</span>
                    </li>
                    <li className="flex items-start gap-3">
                      <ChevronRight className="h-5 w-5 text-primary mt-0.5 shrink-0" />
                      <span><strong className="text-foreground">Performance First:</strong> Real-world benchmarking ensures maximum speed</span>
                    </li>
                    <li className="flex items-start gap-3">
                      <ChevronRight className="h-5 w-5 text-primary mt-0.5 shrink-0" />
                      <span><strong className="text-foreground">Zero Configuration:</strong> No manual setup or hardware-specific code required</span>
                    </li>
                  </ul>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Architecture Overview</h2>
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {[
                        { title: "Detection Layer", desc: "Identifies available hardware (Torch CUDA/MPS, OpenCL) and capabilities" },
                        { title: "Generation Layer", desc: "Creates or selects optimized implementations (Torch, OpenCL, optional AI/Triton)" },
                        { title: "Validation Layer", desc: "Ensures correctness through automated testing with dtype-aware tolerances" },
                        { title: "Cache Layer", desc: "Persists optimal backend decisions and artifacts per device/signature" }
                      ].map((layer) => (
                        <div key={layer.title} className="p-4 rounded-lg bg-muted/50 border border-border/50">
                          <h3 className="font-semibold mb-2">{layer.title}</h3>
                          <p className="text-sm text-muted-foreground">{layer.desc}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </Card>
              </div>
            )}

            {activeSection === "getting-started" && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h1 className="text-4xl font-bold mb-4">Getting Started</h1>
                  <p className="text-lg text-muted-foreground">
                    Quick installation and setup guide
                  </p>
                </div>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Installation</h2>
                  <p className="text-muted-foreground mb-4">Install UHop using pip:</p>
                  <CodeBlock code="pip install uhop" language="bash" />
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Quick Start</h2>
                  <p className="text-muted-foreground mb-4">Use the optimize decorator to accelerate your function:</p>
                  <CodeBlock
                    code={`import numpy as np
from uhop import optimize

# Decorate a baseline implementation (NumPy shown here)
@optimize("matmul")
def matmul_np(A, B):
    return np.array(A) @ np.array(B)

A = np.random.default_rng(0).random((256,256), dtype=np.float32)
B = np.random.default_rng(1).random((256,256), dtype=np.float32)
C = matmul_np(A, B)  # UHOP will choose the best backend and cache it`}
                    language="python"
                  />
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">CLI Usage</h2>
                  <p className="text-muted-foreground mb-4">Core CLI commands shipped with UHOP:</p>
                  <div className="space-y-4">
                    <CodeBlock code="uhop info --json" language="bash" />
                    <CodeBlock code="uhop demo --size 256 --iters 3" language="bash" />
                    <CodeBlock code="uhop demo-conv2d-relu --c-in 3 --c-out 16 --h 64 --w 64 --k 3" language="bash" />
                    <CodeBlock code="uhop cache list" language="bash" />
                    <CodeBlock code="uhop cache show matmul" language="bash" />
                    <CodeBlock code="uhop cache invalidate --device mps" language="bash" />
                    <CodeBlock code="uhop cache invalidate --all" language="bash" />
                    <CodeBlock code="uhop ai-generate --operation matmul --target opencl" language="bash" />
                    <CodeBlock code="uhop ai-generate-fused --target opencl" language="bash" />
                  </div>
                </Card>
              </div>
            )}

            {activeSection === "core-concepts" && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h1 className="text-4xl font-bold mb-4">Core Concepts</h1>
                  <p className="text-lg text-muted-foreground">
                    Understanding UHop's fundamental mechanisms
                  </p>
                </div>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Hardware Detection</h2>
                  <p className="text-muted-foreground mb-4">
                    UHOP automatically discovers available compute devices on your system. It identifies:
                  </p>
                  <ul className="space-y-2 text-muted-foreground ml-6">
                    <li>• CPU architecture and capabilities</li>
                    <li>• NVIDIA/AMD GPUs via Torch CUDA (if present)</li>
                    <li>• Apple GPUs via Torch MPS (Apple Silicon)</li>
                    <li>• OpenCL-compatible devices (Apple, AMD, Intel, etc.)</li>
                    <li>• Optional Triton support (Linux)</li>
                  </ul>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Kernel Selection & Generation</h2>
                  <p className="text-muted-foreground mb-4">
                    For each operation and hardware combination, UHOP can:
                  </p>
                  <ol className="space-y-3 text-muted-foreground ml-6">
                    <li><strong className="text-foreground">1. Prefer accelerators:</strong> Torch CUDA &gt; Torch MPS &gt; Triton (opt) &gt; OpenCL &gt; Torch CPU &gt; NumPy</li>
                    <li><strong className="text-foreground">2. AI generation (opt-in):</strong> Create new kernels using LLM-based code generation (gated by validation)</li>
                    <li><strong className="text-foreground">3. Caching:</strong> Persist winning choice with metadata for instant reuse</li>
                  </ol>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Validation System</h2>
                  <p className="text-muted-foreground mb-4">
                    Every generated or selected kernel undergoes rigorous testing:
                  </p>
                  <CodeBlock
                    code={`from uhop.validation import validate_callable
import numpy as np

def ref(A, B):
    return A @ B

def cand(A, B):
    return np.matmul(A, B)

specs = [{"shape": (64,64), "dtype": np.float32}, {"shape": (64,64), "dtype": np.float32}]
res = validate_callable(cand, ref, specs, strict=True)
print(res.ok, res.max_abs_err, res.max_rel_err)`}
                    language="python"
                  />
                  <p className="text-muted-foreground mt-4">
                    Tip: enable strict validation via CLI <code className="font-mono">--strict-validate</code> or env <code className="font-mono">UHOP_STRICT_VALIDATE=1</code>.
                  </p>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Caching Mechanism</h2>
                  <p className="text-muted-foreground mb-4">
                    UHOP maintains a persistent cache of optimal decisions and artifacts:
                  </p>
                  <ul className="space-y-2 text-muted-foreground ml-6">
                    <li>• Indexed by: operation and optional input signature</li>
                    <li>• Stores: backend, path, kernel metadata</li>
                    <li>• Metadata: <code className="font-mono">device_hint</code>, <code className="font-mono">driver_info</code>, <code className="font-mono">source_hash</code>, <code className="font-mono">_cached_at</code></li>
                    <li>• Invalidate via CLI: <code className="font-mono">uhop cache invalidate --all|--device|--backend</code></li>
                  </ul>
                </Card>
              </div>
            )}

            {activeSection === "cli-usage" && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h1 className="text-4xl font-bold mb-4">CLI Usage</h1>
                  <p className="text-lg text-muted-foreground">
                    Command-line interface reference
                  </p>
                </div>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Hardware and Demos</h2>
                  <p className="text-muted-foreground mb-4">Inspect and run built-in demos:</p>
                  <CodeBlock
                    code={`uhop info --json
uhop demo --size 192 --iters 3
uhop demo-conv2d-relu --n 1 --c-in 3 --c-out 16 --h 64 --w 64 --k 3 --stride 1 --padding 1`}
                    language="bash"
                  />
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Cache Management</h2>
                  <p className="text-muted-foreground mb-4">Manage cached decisions:</p>
                  <CodeBlock
                    code={`uhop cache list
uhop cache show matmul
uhop cache delete matmul
uhop cache clear
uhop cache invalidate --device mps
uhop cache invalidate --backend opencl
uhop cache invalidate --all`}
                    language="bash"
                  />
                </Card>

                <Card className="p-6 bg-card/50">
                </Card>
              </div>
            )}

            {activeSection === "developer-guide" && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h1 className="text-4xl font-bold mb-4">Developer Guide</h1>
                  <p className="text-lg text-muted-foreground">
                    Extending and customizing UHop
                  </p>
                </div>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Using the Decorator</h2>
                  <p className="text-muted-foreground mb-4">Optimize your own function with the built-in decorator:</p>
                  <CodeBlock
                    code={`from uhop import optimize

@optimize("relu")
def relu_np(x):
    return (x > 0) * x

# Usage
import numpy as np
x = np.random.default_rng(0).random(1_000_000, dtype=np.float32)
y = relu_np(x)`}
                    language="python"
                  />
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">AI-Assisted Codegen (Optional)</h2>
                  <p className="text-muted-foreground mb-4">Use the CLI helpers to generate and test candidate kernels:</p>
                  <CodeBlock
                    code={`# Generate OpenCL matmul candidates and run smoke test
uhop ai-generate --operation matmul --target opencl

# Generate fused Conv+ReLU candidates and test
uhop ai-generate-fused --target opencl`}
                    language="bash"
                  />
                  <p className="text-muted-foreground mt-2">
                    Safety: AI kernels are only adopted if they pass validation. Enable <code className="font-mono">--strict-validate</code> to tighten tolerances.
                  </p>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Torch MPS Convenience</h2>
                  <p className="text-muted-foreground mb-4">Explicitly use Apple MPS via the MPS facade:</p>
                  <CodeBlock
                    code={`from uhop.backends import is_mps_available, mps_matmul
import torch

if is_mps_available():
    a = torch.randn(256,256, device="mps")
    b = torch.randn(256,256, device="mps")
    c = mps_matmul(a, b)  # stays on MPS
`}
                    language="python"
                  />
                </Card>
              </div>
            )}

            {activeSection === "api-reference" && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h1 className="text-4xl font-bold mb-4">API Reference</h1>
                  <p className="text-lg text-muted-foreground">
                    Complete Python API documentation
                  </p>
                </div>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">optimize()</h2>
                  <p className="text-muted-foreground mb-4">Decorator to optimize a function by operation key</p>
                  <CodeBlock
                    code={`from uhop import optimize

@optimize("matmul")
def matmul_np(A, B):
    ...`}
                    language="python"
                  />
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">UHopOptimizer</h2>
                  <p className="text-muted-foreground mb-4">Programmatic control of backend policy</p>
                  <CodeBlock
                    code={`from uhop import UHopOptimizer
opt = UHopOptimizer()
# opt.optimize_fn(...) - see repository examples for usage`}
                    language="python"
                  />
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">detect_hardware()</h2>
                  <p className="text-muted-foreground mb-4">Snapshot of hardware for decisions and docs</p>
                  <CodeBlock
                    code={`from uhop import detect_hardware
hw = detect_hardware()
print(hw.vendor, hw.kind, hw.name)`}
                    language="python"
                  />
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">UhopCache</h2>
                  <p className="text-muted-foreground mb-4">Inspect and manage cache programmatically</p>
                  <CodeBlock
                    code={`from uhop import UhopCache
c = UhopCache()
print(c.all())
c.invalidate_device("mps")  # remove MPS-specific entries
`}
                    language="python"
                  />
                </Card>
              </div>
            )}
          </main>
        </div>
      </div>
    </div>
  );
};

export default Docs;

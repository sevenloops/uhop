import { useState } from "react";
import { Link } from "react-router-dom";
import {
  ChevronRight,
  Book,
  Rocket,
  Code,
  Cog,
  FileText,
  Zap,
} from "lucide-react";
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
    { id: "opencl-clblast", title: "OpenCL & CLBlast", icon: Zap },
    { id: "autotune-tuning", title: "Autotune & Tuning", icon: Cog },
    { id: "troubleshooting", title: "Troubleshooting", icon: FileText },
    { id: "resources", title: "Resources", icon: Book },
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
            {/* Mobile nav (chips) */}
            <div className="lg:hidden sticky top-16 z-10 -mt-4 mb-6 bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
              <div className="flex gap-2 overflow-x-auto px-1 py-3">
                {sections.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => setActiveSection(s.id)}
                    className={`shrink-0 rounded-full border px-3 py-1.5 text-sm transition-colors ${
                      activeSection === s.id
                        ? "bg-primary text-primary-foreground border-primary"
                        : "bg-muted text-foreground hover:bg-muted/80 border-border"
                    }`}
                  >
                    {s.title}
                  </button>
                ))}
              </div>
            </div>
            {activeSection === "introduction" && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h1 className="text-4xl font-bold mb-4">
                    Introduction to UHOP
                  </h1>
                  <p className="text-lg text-muted-foreground">
                    Universal Hardware Optimization Protocol
                  </p>
                </div>

                <Card className="p-6 bg-card/50 border-primary/20">
                  <h2 className="text-2xl font-semibold mb-4">What is UHOP?</h2>
                  <p className="text-muted-foreground leading-relaxed mb-4">
                    UHOP is an AI-driven hardware optimization runtime that
                    automatically detects your computing hardware, prioritizes
                    accelerators (CUDA, Apple MPS, OpenCL) when available,
                    generates or selects optimized implementations for core ops,
                    validates correctness, benchmarks their performance, and
                    caches the fastest choices for reuse.
                  </p>
                  <p className="text-muted-foreground leading-relaxed">
                    It provides a universal abstraction layer across different
                    hardware backends (Torch CUDA/MPS/CPU, OpenCL, optional
                    Triton), enabling developers to write once and run optimally
                    everywhere.
                  </p>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">
                    Why UHop Matters
                  </h2>
                  <ul className="space-y-3 text-muted-foreground">
                    <li className="flex items-start gap-3">
                      <ChevronRight className="h-5 w-5 text-primary mt-0.5 shrink-0" />
                      <span>
                        <strong className="text-foreground">
                          Hardware Agnostic:
                        </strong>{" "}
                        Write code once, optimize for any device automatically
                      </span>
                    </li>
                    <li className="flex items-start gap-3">
                      <ChevronRight className="h-5 w-5 text-primary mt-0.5 shrink-0" />
                      <span>
                        <strong className="text-foreground">AI-Powered:</strong>{" "}
                        Leverages machine learning to generate and validate
                        optimal kernels
                      </span>
                    </li>
                    <li className="flex items-start gap-3">
                      <ChevronRight className="h-5 w-5 text-primary mt-0.5 shrink-0" />
                      <span>
                        <strong className="text-foreground">
                          Performance First:
                        </strong>{" "}
                        Real-world benchmarking ensures maximum speed
                      </span>
                    </li>
                    <li className="flex items-start gap-3">
                      <ChevronRight className="h-5 w-5 text-primary mt-0.5 shrink-0" />
                      <span>
                        <strong className="text-foreground">
                          Zero Configuration:
                        </strong>{" "}
                        No manual setup or hardware-specific code required
                      </span>
                    </li>
                  </ul>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">
                    Architecture Overview
                  </h2>
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {[
                        {
                          title: "Detection Layer",
                          desc: "Identifies available hardware (Torch CUDA/MPS, OpenCL) and capabilities",
                        },
                        {
                          title: "Generation Layer",
                          desc: "Creates or selects optimized implementations (Torch, OpenCL, optional AI/Triton)",
                        },
                        {
                          title: "Validation Layer",
                          desc: "Ensures correctness through automated testing with dtype-aware tolerances",
                        },
                        {
                          title: "Cache Layer",
                          desc: "Persists optimal backend decisions and artifacts per device/signature",
                        },
                      ].map((layer) => (
                        <div
                          key={layer.title}
                          className="p-4 rounded-lg bg-muted/50 border border-border/50"
                        >
                          <h3 className="font-semibold mb-2">{layer.title}</h3>
                          <p className="text-sm text-muted-foreground">
                            {layer.desc}
                          </p>
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
                  <p className="text-muted-foreground mb-4">
                    Install UHop using pip:
                  </p>
                  <CodeBlock code="pip install uhop" language="bash" />
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Quick Start</h2>
                  <p className="text-muted-foreground mb-4">
                    Use the optimize decorator to accelerate your function:
                  </p>
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
                  <p className="text-muted-foreground mb-4">
                    Core CLI commands shipped with UHOP:
                  </p>
                  <div className="space-y-4">
                    <CodeBlock code="uhop info --json" language="bash" />
                    <CodeBlock
                      code="uhop demo --size 256 --iters 3"
                      language="bash"
                    />
                    <CodeBlock
                      code="uhop demo-conv2d-relu --c-in 3 --c-out 16 --h 64 --w 64 --k 3"
                      language="bash"
                    />
                    <CodeBlock code="uhop cache list" language="bash" />
                    <CodeBlock code="uhop cache show matmul" language="bash" />
                    <CodeBlock
                      code="uhop cache invalidate --device mps"
                      language="bash"
                    />
                    <CodeBlock
                      code="uhop cache invalidate --all"
                      language="bash"
                    />
                    <CodeBlock
                      code="uhop ai-generate --operation matmul --target opencl"
                      language="bash"
                    />
                    <CodeBlock
                      code="uhop ai-generate-fused --target opencl"
                      language="bash"
                    />
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
                  <h2 className="text-2xl font-semibold mb-4">
                    Hardware Detection
                  </h2>
                  <p className="text-muted-foreground mb-4">
                    UHOP automatically discovers available compute devices on
                    your system. It identifies:
                  </p>
                  <ul className="space-y-2 text-muted-foreground ml-6">
                    <li>• CPU architecture and capabilities</li>
                    <li>• NVIDIA/AMD GPUs via Torch CUDA (if present)</li>
                    <li>• Apple GPUs via Torch MPS (Apple Silicon)</li>
                    <li>
                      • OpenCL-compatible devices (Apple, AMD, Intel, etc.)
                    </li>
                    <li>• Optional Triton support (Linux)</li>
                  </ul>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">
                    Kernel Selection & Generation
                  </h2>
                  <p className="text-muted-foreground mb-4">
                    For each operation and hardware combination, UHOP can:
                  </p>
                  <ol className="space-y-3 text-muted-foreground ml-6">
                    <li>
                      <strong className="text-foreground">
                        1. Prefer accelerators:
                      </strong>{" "}
                      Torch CUDA &gt; Torch MPS &gt; Triton (opt) &gt; OpenCL
                      &gt; Torch CPU &gt; NumPy
                    </li>
                    <li>
                      <strong className="text-foreground">
                        2. AI generation (opt-in):
                      </strong>{" "}
                      Create new kernels using LLM-based code generation (gated
                      by validation)
                    </li>
                    <li>
                      <strong className="text-foreground">3. Caching:</strong>{" "}
                      Persist winning choice with metadata for instant reuse
                    </li>
                  </ol>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">
                    Validation System
                  </h2>
                  <p className="text-muted-foreground mb-4">
                    Every generated or selected kernel undergoes rigorous
                    testing:
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
                    Tip: enable strict validation via CLI{" "}
                    <code className="font-mono">--strict-validate</code> or env{" "}
                    <code className="font-mono">UHOP_STRICT_VALIDATE=1</code>.
                  </p>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">
                    Caching Mechanism
                  </h2>
                  <p className="text-muted-foreground mb-4">
                    UHOP maintains a persistent cache of optimal decisions and
                    artifacts:
                  </p>
                  <ul className="space-y-2 text-muted-foreground ml-6">
                    <li>
                      • Indexed by: operation and optional input signature
                    </li>
                    <li>• Stores: backend, path, kernel metadata</li>
                    <li>
                      • Metadata: <code className="font-mono">device_hint</code>
                      , <code className="font-mono">driver_info</code>,{" "}
                      <code className="font-mono">source_hash</code>,{" "}
                      <code className="font-mono">_cached_at</code>
                    </li>
                    <li>
                      • Invalidate via CLI:{" "}
                      <code className="font-mono">
                        uhop cache invalidate --all|--device|--backend
                      </code>
                    </li>
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
                  <h2 className="text-2xl font-semibold mb-4">
                    Hardware and Demos
                  </h2>
                  <p className="text-muted-foreground mb-4">
                    Inspect and run built-in demos:
                  </p>
                  <CodeBlock
                    code={`uhop info --json
uhop demo --size 192 --iters 3
uhop demo-conv2d-relu --n 1 --c-in 3 --c-out 16 --h 64 --w 64 --k 3 --stride 1 --padding 1`}
                    language="bash"
                  />
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">
                    Cache Management
                  </h2>
                  <p className="text-muted-foreground mb-4">
                    Manage cached decisions:
                  </p>
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
                  <h2 className="text-2xl font-semibold mb-4">
                    Autotune & CLBlast Flags
                  </h2>
                  <p className="text-muted-foreground mb-3">
                    UHOP persists simple autotune metadata and device flags
                    (e.g., marking CLBlast as unstable on a device) in
                    <code className="font-mono">
                      {" "}
                      ~/.uhop_mvp_cache/autotune.json
                    </code>
                    .
                  </p>
                  <div className="space-y-3">
                    <p className="text-sm text-muted-foreground">
                      Inspect CLBlast unstable devices in{" "}
                      <code className="font-mono">uhop info --json</code>{" "}
                      output:
                    </p>
                    <CodeBlock
                      code={`uhop info --json | jq '.opencl_clblast_unstable_devices'`}
                      language="bash"
                    />
                    <p className="text-sm text-muted-foreground">
                      Clear the CLBlast unstable flag (all devices, or filter by
                      device substring):
                    </p>
                    <CodeBlock
                      code={`# Clear all CLBlast unstable flags
uhop cache autotune clear-clblast-unstable

# Filter by device name substring (case-insensitive)
uhop cache autotune clear-clblast-unstable gfx

# Exact match and dry-run preview
uhop cache autotune clear-clblast-unstable "Radeon RX 6800" --exact --dry-run`}
                      language="bash"
                    />
                  </div>
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
                  <h2 className="text-2xl font-semibold mb-4">
                    Using the Decorator
                  </h2>
                  <p className="text-muted-foreground mb-4">
                    Optimize your own function with the built-in decorator:
                  </p>
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
                  <h2 className="text-2xl font-semibold mb-4">
                    AI-Assisted Codegen (Optional)
                  </h2>
                  <p className="text-muted-foreground mb-4">
                    Use the CLI helpers to generate and test candidate kernels:
                  </p>
                  <CodeBlock
                    code={`# Generate OpenCL matmul candidates and run smoke test
uhop ai-generate --operation matmul --target opencl

# Generate fused Conv+ReLU candidates and test
uhop ai-generate-fused --target opencl`}
                    language="bash"
                  />
                  <p className="text-muted-foreground mt-2">
                    Safety: AI kernels are only adopted if they pass validation.
                    Enable <code className="font-mono">--strict-validate</code>{" "}
                    to tighten tolerances.
                  </p>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">
                    OpenCL + CLBlast (Optional)
                  </h2>
                  <p className="text-muted-foreground mb-4">
                    UHOP can use{" "}
                    <a
                      className="underline"
                      href="https://github.com/CNugteren/CLBlast"
                      target="_blank"
                      rel="noreferrer noopener"
                    >
                      CLBlast
                    </a>{" "}
                    for GEMM to power matmul and the im2col+GEMM Conv2D path.
                  </p>
                  <div className="space-y-3">
                    <p className="text-sm text-muted-foreground">
                      Environment knobs:
                    </p>
                    <CodeBlock
                      code={`# Point UHOP to your CLBlast shared library (DLL/SO/Dylib)
export CLBLAST_LIBRARY=/abs/path/to/clblast.dll

# Prefer CLBlast for matmul
export UHOP_OPENCL_MATMUL_IMPL=clblast

# Prefer im2col+GEMM for Conv2D (requires CLBlast)
export UHOP_OPENCL_CONV_IMPL=im2col_gemm

# Optional compile-time vectorization hints for OpenCL kernels
export UHOP_OPENCL_VEC_CANDIDATES="1,4"`}
                      language="bash"
                    />
                    <p className="text-sm text-muted-foreground">
                      Troubleshooting (Windows): UHOP uses a robust ctypes
                      wrapper (WinDLL and a non-NULL cl_event*). If you still
                      see an access violation in
                      <code className="font-mono"> CLBlastSgemm</code>, keep
                      working with the stable tiled kernels by setting
                      <code className="font-mono">
                        {" "}
                        UHOP_OPENCL_CONV_IMPL=tiled
                      </code>{" "}
                      and/or
                      <code className="font-mono">
                        {" "}
                        UHOP_OPENCL_MATMUL_IMPL=tiled
                      </code>
                      , and consider updating your OpenCL drivers or trying a
                      different CLBlast build. See the repository docs for full
                      details.
                    </p>
                  </div>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">
                    Torch MPS Convenience
                  </h2>
                  <p className="text-muted-foreground mb-4">
                    Explicitly use Apple MPS via the MPS facade:
                  </p>
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

            {activeSection === "opencl-clblast" && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h1 className="text-4xl font-bold mb-4">OpenCL & CLBlast</h1>
                  <p className="text-lg text-muted-foreground">
                    Detailed configuration and behavior
                  </p>
                </div>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">
                    How UHOP uses CLBlast
                  </h2>
                  <ul className="space-y-2 text-muted-foreground ml-6">
                    <li>
                      • Matmul: If{" "}
                      <code className="font-mono">
                        UHOP_OPENCL_MATMUL_IMPL=clblast
                      </code>
                      , UHOP will dispatch GEMM to CLBlast SGEMM.
                    </li>
                    <li>
                      • Conv2D: If{" "}
                      <code className="font-mono">
                        UHOP_OPENCL_CONV_IMPL=im2col_gemm
                      </code>{" "}
                      (or in <em>auto</em> mode and heuristic selects GEMM),
                      UHOP runs an OpenCL <code>im2col</code> kernel and
                      multiplies via CLBlast.
                    </li>
                    <li>
                      • Heuristic: <em>auto</em> prefers tiled for small shapes,
                      prefers im2col+GEMM for large kernels/output, and always
                      falls back if CLBlast is unavailable or unstable.
                    </li>
                  </ul>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Configuration</h2>
                  <CodeBlock
                    code={`# Required when CLBlast is not in the default search path
export CLBLAST_LIBRARY=/abs/path/to/clblast.(dll|so|dylib)

# Force implementations
export UHOP_OPENCL_MATMUL_IMPL=(tiled|clblast)
export UHOP_OPENCL_CONV_IMPL=(auto|tiled|im2col_gemm)

# Vectorization candidates for OpenCL kernels (compile-time defines)
export UHOP_OPENCL_VEC_CANDIDATES="1,2,4,8"`}
                    language="bash"
                  />
                  <p className="text-muted-foreground mt-3">
                    Note: vectorization factors &gt;1 require matching alignment
                    and kernel support; UHOP explores candidates conservatively.
                  </p>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">
                    Unstable Device Flag
                  </h2>
                  <p className="text-muted-foreground mb-3">
                    If CLBlast raises a runtime error (e.g., access violation on
                    some Windows setups), UHOP records a per-device
                    <em>unstable</em> flag to avoid repeated attempts in{" "}
                    <em>auto</em> mode.
                  </p>
                  <div className="space-y-3">
                    <p className="text-sm text-muted-foreground">
                      Where it’s stored:
                    </p>
                    <CodeBlock
                      code={`~/.uhop_mvp_cache/autotune.json
# key format: opencl|clblast|sgemm|<device_name>|device
{"unstable": true}`}
                      language="json"
                    />
                    <p className="text-sm text-muted-foreground">
                      How to view from CLI:
                    </p>
                    <CodeBlock
                      code={`uhop info --json | jq '.opencl_clblast_unstable_devices'`}
                      language="bash"
                    />
                    <p className="text-sm text-muted-foreground">
                      How to clear (allow re-test):
                    </p>
                    <CodeBlock
                      code={`uhop cache autotune clear-clblast-unstable
uhop cache autotune clear-clblast-unstable gfx  # substring filter
uhop cache autotune clear-clblast-unstable "Radeon" --exact`}
                      language="bash"
                    />
                  </div>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Windows Notes</h2>
                  <ul className="space-y-2 text-muted-foreground ml-6">
                    <li>
                      • UHOP uses <code className="font-mono">WinDLL</code> and
                      a non-NULL <code className="font-mono">cl_event*</code> in
                      the ctypes binding.
                    </li>
                    <li>
                      • If you still encounter access violations, prefer the
                      tiled implementations and consider trying different
                      CLBlast builds or updating your OpenCL driver.
                    </li>
                    <li>
                      • The unstable flag ensures <em>auto</em> skips CLBlast to
                      avoid repeated fallback overhead; you can clear it to
                      re-test.
                    </li>
                  </ul>
                </Card>
              </div>
            )}

            {activeSection === "autotune-tuning" && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h1 className="text-4xl font-bold mb-4">Autotune & Tuning</h1>
                  <p className="text-lg text-muted-foreground">
                    How tuning works and how to inspect it
                  </p>
                </div>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">
                    What UHOP Tunes
                  </h2>
                  <ul className="space-y-2 text-muted-foreground ml-6">
                    <li>
                      • Matmul tiled: local sizes (tile), compile-time defines{" "}
                      <code className="font-mono">TILE</code> and{" "}
                      <code className="font-mono">VEC</code>.
                    </li>
                    <li>
                      • Conv2D tiled: <code className="font-mono">TILE_W</code>,{" "}
                      <code className="font-mono">TILE_H</code>, and optional{" "}
                      <code className="font-mono">VEC</code>.
                    </li>
                    <li>
                      • Fused Conv2D+ReLU: best local size from a small
                      candidate set.
                    </li>
                  </ul>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">
                    Persistence Format
                  </h2>
                  <p className="text-muted-foreground mb-3">
                    Stored in{" "}
                    <code className="font-mono">
                      ~/.uhop_mvp_cache/autotune.json
                    </code>{" "}
                    with shape-keyed entries:
                  </p>
                  <CodeBlock
                    code={`# Example keys
opencl|matmul|matmul_tiled|<device>|M<M>_K<K>_N<N>
opencl|conv2d|conv2d_tiled|<device>|N<N>_C<C>_H<H>_W<W>_Co<Co>_KH<KH>_KW<KW>_S<S>_P<P>

# Values include tuned local sizes and compile-time parameters
{"lsz": [16, 16], "tile": 16, "vec": 1}`}
                    language="json"
                  />
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Tips</h2>
                  <ul className="space-y-2 text-muted-foreground ml-6">
                    <li>
                      • To explore vectorization, set{" "}
                      <code className="font-mono">
                        UHOP_OPENCL_VEC_CANDIDATES
                      </code>{" "}
                      (e.g., <code className="font-mono">"1,4"</code>).
                    </li>
                    <li>
                      • Autotune runs quickly by timing a small candidate set;
                      persisted choices avoid re-tuning.
                    </li>
                    <li>
                      • You can remove tuned entries with{" "}
                      <code className="font-mono">uhop cache invalidate</code>{" "}
                      or by editing{" "}
                      <code className="font-mono">autotune.json</code>.
                    </li>
                  </ul>
                </Card>
              </div>
            )}

            {activeSection === "troubleshooting" && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h1 className="text-4xl font-bold mb-4">Troubleshooting</h1>
                  <p className="text-lg text-muted-foreground">
                    Common issues and fixes
                  </p>
                </div>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">
                    OpenCL / CLBlast
                  </h2>
                  <ul className="space-y-2 text-muted-foreground ml-6">
                    <li>
                      • Library not found: set{" "}
                      <code className="font-mono">CLBLAST_LIBRARY</code> to the
                      absolute path of the shared library.
                    </li>
                    <li>
                      • Access violation on Windows: stick to tiled
                      implementations and clear the unstable flag if you want to
                      re-test later.
                    </li>
                    <li>
                      • Wrong device: use{" "}
                      <code className="font-mono">
                        UHOP_OPENCL_DEVICE_INDEX
                      </code>{" "}
                      or CLI options that accept{" "}
                      <code className="font-mono">--ocl-device</code>.
                    </li>
                  </ul>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Validation</h2>
                  <ul className="space-y-2 text-muted-foreground ml-6">
                    <li>
                      • Use <code className="font-mono">--strict-validate</code>{" "}
                      to tighten tolerances when adopting new kernels.
                    </li>
                    <li>
                      • For reproducibility, keep shapes/dtypes consistent
                      between reference and candidate functions.
                    </li>
                  </ul>
                </Card>
              </div>
            )}

            {activeSection === "resources" && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h1 className="text-4xl font-bold mb-4">Resources</h1>
                  <p className="text-lg text-muted-foreground">
                    Further reading, articles, and links
                  </p>
                </div>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">Articles</h2>
                  <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
                    <li>
                      Introducing UHOP — An Open Hardware Optimization Platform
                      for GPU Compute &nbsp;
                      <a
                        href="https://medium.com/@danbis664/introducing-uhop-an-open-hardware-optimization-platform-for-gpu-compute-072420544812?postPublishedType=initial"
                        target="_blank"
                        rel="noreferrer noopener"
                        className="underline"
                      >
                        Read on Medium
                      </a>
                    </li>
                  </ul>
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">
                    Repositories & Docs
                  </h2>
                  <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
                    <li>
                      CLBlast project: &nbsp;
                      <a
                        href="https://github.com/CNugteren/CLBlast"
                        target="_blank"
                        rel="noreferrer noopener"
                        className="underline"
                      >
                        github.com/CNugteren/CLBlast
                      </a>
                    </li>
                    <li>
                      Check the repo docs for build notes and platform-specific
                      guidance (e.g., Windows drivers, DLL search path).
                    </li>
                  </ul>
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
                  <p className="text-muted-foreground mb-4">
                    Decorator to optimize a function by operation key
                  </p>
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
                  <p className="text-muted-foreground mb-4">
                    Programmatic control of backend policy
                  </p>
                  <CodeBlock
                    code={`from uhop import UHopOptimizer
opt = UHopOptimizer()
# opt.optimize_fn(...) - see repository examples for usage`}
                    language="python"
                  />
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">
                    detect_hardware()
                  </h2>
                  <p className="text-muted-foreground mb-4">
                    Snapshot of hardware for decisions and docs
                  </p>
                  <CodeBlock
                    code={`from uhop import detect_hardware
hw = detect_hardware()
print(hw.vendor, hw.kind, hw.name)`}
                    language="python"
                  />
                </Card>

                <Card className="p-6 bg-card/50">
                  <h2 className="text-2xl font-semibold mb-4">UhopCache</h2>
                  <p className="text-muted-foreground mb-4">
                    Inspect and manage cache programmatically
                  </p>
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

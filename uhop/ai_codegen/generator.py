# uhop/ai_codegen/generator.py
"""
Consolidated AI Codegen provider.

- Primary provider: OpenAI (ChatCompletion)
- Optional fallback: DeepSeek provider (if DEEPSEEK_API_KEY and DEEPSEEK_URL are set)
- Supports generating CUDA C code or Python kernels depending on requested target.
- Saves generated code into uhop/generated_kernels/
"""
import os
import re
import ast
from pathlib import Path
from typing import Optional

# Optional provider: deepseek
try:
    from . import deepseek_provider  # type: ignore
except Exception:
    deepseek_provider = None

try:
    # Prefer OpenAI v1 SDK usage
    from openai import OpenAI as _OpenAIClient  # type: ignore
    _OPENAI_V1 = True
except Exception:
    _OpenAIClient = None
    _OPENAI_V1 = False
    try:
        import openai  # type: ignore
    except Exception:
        openai = None  # type: ignore

GENERATED_DIR = Path(__file__).resolve().parent.parent / "generated_kernels"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

def _extract_code_blocks(text: str):
    blocks = re.findall(r"```(?:opencl|cuda|c\+\+|c|cpp|python|)\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    # fallback heuristics
    if 'extern "C"' in text or 'kernel' in text:
        return [text.strip()]
    return []

def _verify_syntax_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

class AICodegen:
    def __init__(self, model: str = os.environ.get("UHOP_OPENAI_MODEL", "gpt-4o-mini")):
        self.model = model
        self.last_prompt: Optional[str] = None

    def _call_openai(self, prompt: str, max_tokens: int = 1200, temperature: float = 0.0):
        # Requires OPENAI_API_KEY; supports both legacy and v1 SDKs.
        if "OPENAI_API_KEY" not in os.environ:
            return None
        try:
            if _OPENAI_V1 and _OpenAIClient is not None:
                client = _OpenAIClient()
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role":"system","content":"You are an assistant that outputs runnable kernels (CUDA C, OpenCL C, Triton, or Python)."},
                        {"role":"user","content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return resp.choices[0].message.content
            # Legacy fallback
            if 'openai' in globals() and openai is not None:
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role":"system","content":"You are an assistant that outputs runnable kernels (CUDA C, OpenCL C, Triton, or Python)."},
                        {"role":"user","content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return resp["choices"][0]["message"]["content"]
        except Exception:
            return None
        return None

    def _call_deepseek(self, prompt: str):
        if deepseek_provider is None or not getattr(deepseek_provider, "is_configured", lambda: False)():
            return None
        try:
            return deepseek_provider.generate(prompt)
        except Exception:
            return None

    def generate(self, operation_name: str, target: Optional[str] = None, prompt_extra: Optional[str] = None, *, temperature: float = 0.0, suffix: Optional[str] = None) -> Path:
        """
        Generate a kernel for operation_name.
        target: "cuda", "opencl", "python", "triton" (advisory; generator prefers CUDA).
        Returns path to saved code file in generated_kernels/.
        """
        # Build a flexible prompt depending on target
        if target == "opencl":
            op = operation_name.lower()
            if op == "matmul":
                prompt = (
                    "Produce a single OpenCL C kernel for operation matmul.\n"
                    "Name the kernel generated_matmul. Use this exact signature:\n"
                    "__kernel void generated_matmul(const int M, const int N, const int K,\n"
                    "                                  __global const float* A,\n"
                    "                                  __global const float* B,\n"
                    "                                  __global float* C);\n"
                    "Implement C = A (MxK) x B (KxN).\n"
                    "- Use simple global indexing; no extensions; no includes; compile on OpenCL 2.0.\n"
                    "- Bounds check all reads/writes; do not assume multiples of workgroup sizes.\n"
                    "Output only the code block."
                )
            elif op == "relu":
                prompt = (
                    "Produce a single OpenCL C kernel for ReLU.\n"
                    "Name the kernel generated_relu. Signature:\n"
                    "__kernel void generated_relu(const int N, __global const float* X, __global float* Y);\n"
                    "Compute Y[i] = max(X[i], 0). Bounds-check global id. Output only the code block."
                )
            elif op == "conv2d":
                prompt = (
                    "Produce an OpenCL C kernel for NCHW Conv2D. Name it generated_conv2d.\n"
                    "Signature:\n"
                    "__kernel void generated_conv2d(const int N, const int C_in, const int H, const int W,\n"
                    "                                 const int C_out, const int KH, const int KW,\n"
                    "                                 const int stride, const int pad,\n"
                    "                                 __global const float* input,  // N*C_in*H*W\n"
                    "                                 __global const float* weight, // C_out*C_in*KH*KW\n"
                    "                                 __global float* output,       // N*C_out*outH*outW\n"
                    "                                 const int outH, const int outW);\n"
                    "Use 3D NDRange (x=outW, y=outH, z=N*C_out) with bounds checks. Output only the code block."
                )
            else:
                prompt = f"Produce an OpenCL C kernel for operation {operation_name}. Output only the code block."
        elif target == "triton":
            prompt = f"Produce a Triton Python kernel for operation {operation_name} using triton.jit. Output only the code block."
        elif target == "python":
            prompt = f"Produce a single Python function named generated_{operation_name}(a, b) that implements the operation using numpy. Output only the code block."
        else:
            # default: CUDA
            prompt = f'Produce a CUDA C kernel named {operation_name}_kernel with signature: extern "C" __global__ void {operation_name}_kernel(const float* A, const float* B, float* C, int N, int M, int K). Output only the code block.'
        if prompt_extra:
            prompt += "\n\n" + prompt_extra

        # Try OpenAI first
        self.last_prompt = prompt
        text = self._call_openai(prompt, temperature=temperature)
        provider = "openai"
        if not text:
            # fallback to deepseek
            text = self._call_deepseek(prompt)
            provider = "deepseek" if text else None

        if not text:
            raise RuntimeError("No AI provider produced code. Configure OPENAI_API_KEY or DeepSeek credentials.")

        blocks = _extract_code_blocks(text)
        if not blocks:
            # fallback to whole response
            blocks = [text.strip()]

        code = max(blocks, key=len).strip()

        # Determine extension
        ext = ".cu" if target in (None, "cuda") else ".cl" if target == "opencl" else ".py" if target == "python" else ".py"
        suffix = suffix or ""
        filename = GENERATED_DIR / f"ai_{operation_name}{suffix}{ext}"
        # If python target, verify syntax
        if ext == ".py":
            if "import numpy" not in code and "np." in code:
                code = "import numpy as np\n\n" + code
            if not _verify_syntax_python(code):
                raise RuntimeError("AI-generated Python code failed syntax check.")
        # write file
        filename.write_text(code)
        return filename

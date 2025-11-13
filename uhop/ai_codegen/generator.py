# uhop/ai_codegen/generator.py
"""
Consolidated AI code generation providers.

- Providers: OpenAI, DeepSeek
- Supports generating CUDA/OpenCL/Triton/Python kernels depending on requested target.
- Saves generated code into uhop/generated_kernels/
"""

import ast
import os
import re
from pathlib import Path
from typing import Optional

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

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade when requests missing
    requests = None  # type: ignore

GENERATED_DIR = Path(__file__).resolve().parent.parent / "generated_kernels"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SYSTEM_PROMPT = "You are an assistant that outputs runnable kernels (CUDA C, OpenCL C, Triton, or Python)."
DEFAULT_DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def _extract_code_blocks(text: str):
    blocks = re.findall(
        r"```(?:opencl|cuda|c\+\+|c|cpp|python|)\n(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    # fallback heuristics
    if 'extern "C"' in text or "kernel" in text:
        return [text.strip()]
    return []


def _verify_syntax_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


class AICodegen:
    def __init__(self, model: Optional[str] = None, provider: str = "openai"):
        """AI Code generator capable of talking to multiple providers."""

        self.provider = (provider or "openai").lower()
        if model is None:
            if self.provider == "deepseek":
                model = os.environ.get("UHOP_DEEPSEEK_MODEL", "deepseek-coder")
            else:
                model = os.environ.get("UHOP_OPENAI_MODEL", "gpt-4o-mini")
        self.model = model
        self.last_prompt: Optional[str] = None
        self.last_error: Optional[str] = None  # populated if a provider call fails

    def _debug(self) -> bool:
        return os.environ.get("UHOP_AI_DEBUG", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _call_openai(self, prompt: str, max_tokens: int = 1200, temperature: float = 0.0):
        # Requires OPENAI_API_KEY; supports both legacy and v1 SDKs.
        if "OPENAI_API_KEY" not in os.environ:
            self.last_error = "OPENAI_API_KEY not set in environment"
            return None
        try:
            if _OPENAI_V1 and _OpenAIClient is not None:
                client = _OpenAIClient()
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": DEFAULT_SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content
            # Legacy fallback (openai<1.0 style) retained just in case
            if "openai" in globals() and openai is not None:
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": DEFAULT_SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp["choices"][0]["message"]["content"]
            self.last_error = "OpenAI SDK not importable"
        except Exception as e:
            # Capture full error for surfacing later
            self.last_error = f"OpenAI request failed: {type(e).__name__}: {e}"
            if self._debug():
                import traceback

                traceback.print_exc()
        return None

    def _call_deepseek(self, prompt: str, *, max_tokens: int = 1200, temperature: float = 0.0):
        if requests is None:
            self.last_error = "DeepSeek provider requires the 'requests' package"
            return None
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            self.last_error = "DEEPSEEK_API_KEY not set in environment"
            return None
        url = os.environ.get("DEEPSEEK_API_BASE", DEFAULT_DEEPSEEK_URL)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model or "deepseek-coder",
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": max(min(temperature, 1.0), 0.0),
            "max_tokens": max_tokens,
            "stream": False,
        }
        timeout_s = float(os.environ.get("DEEPSEEK_TIMEOUT", "45"))
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # pragma: no cover - network errors
            self.last_error = f"DeepSeek request failed: {type(e).__name__}: {e}"
            if self._debug():
                import traceback

                traceback.print_exc()
            return None

        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            self.last_error = "DeepSeek response missing message content"
            return None

    def generate(
        self,
        operation_name: str,
        target: Optional[str] = None,
        prompt_extra: Optional[str] = None,
        *,
        temperature: float = 0.0,
        suffix: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Path:
        """
        Generate a kernel for operation_name.
        target: "cuda", "opencl", "python", "triton" (advisory; generator prefers CUDA).
        Returns path to saved code file in generated_kernels/.
        """
        # Choose target if not provided based on detected hardware
        if target is None:
            try:
                from ..hardware import detect_hardware

                hw = detect_hardware()
                if hw.kind.startswith("opencl") or hw.vendor in ("amd", "intel"):
                    target = "opencl"
                elif hw.vendor == "nvidia":
                    target = "cuda"
                else:
                    target = "opencl"
            except Exception:
                target = "opencl"

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
        resolved_provider = (provider or self.provider or "openai").lower()
        if resolved_provider == "deepseek":
            text = self._call_deepseek(prompt, max_tokens=1200, temperature=temperature)
        else:
            text = self._call_openai(prompt, max_tokens=1200, temperature=temperature)

        if not text:
            # Surface more helpful diagnostics
            detail = self.last_error or "unknown error"
            raise RuntimeError(
                "AI code generation failed. "
                f"Provider='{resolved_provider}', model='{self.model}'. Detail: {detail}. "
                "Ensure the provider API key is configured and the model name is correct."
            )

        blocks = _extract_code_blocks(text)
        if not blocks:
            # fallback to whole response
            blocks = [text.strip()]

        code = max(blocks, key=len).strip()

        # Determine extension
        ext = (
            ".cu"
            if target in (None, "cuda")
            else ".cl" if target == "opencl" else ".py" if target == "python" else ".py"
        )
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

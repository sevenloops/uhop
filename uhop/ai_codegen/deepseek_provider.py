"""
DeepSeek API provider for AI kernel generation.

Supports both chat completion and code generation endpoints.
"""

import os
from typing import Optional

import requests


def _call_deepseek_api(
    prompt: str, model: str = "deepseek-coder", max_tokens: int = 1200, temperature: float = 0.0
) -> Optional[str]:
    """
    Call DeepSeek API for code generation.

    Requires DEEPSEEK_API_KEY environment variable.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")

    # DeepSeek API endpoint
    url = "https://api.deepseek.com/chat/completions"

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert GPU kernel engineer. Generate efficient, correct kernels for various GPU architectures (CUDA, OpenCL, HIP, Metal). Always output compilable code in code blocks.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"DeepSeek API request failed: {e}")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response format from DeepSeek API: {e}")


class DeepSeekProvider:
    """DeepSeek API provider for kernel generation."""

    def __init__(self, model: Optional[str] = None):
        """
        Initialize DeepSeek provider.

        Parameters
        ----------
        model: str
            DeepSeek model to use. Defaults to "deepseek-coder".
            Override via DEEPSEEK_MODEL env var.
        """
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-coder")
        self.last_error: Optional[str] = None

    def generate(self, prompt: str, max_tokens: int = 1200, temperature: float = 0.0) -> str:
        """
        Generate kernel code using DeepSeek API.

        Parameters
        ----------
        prompt: str
            The prompt for kernel generation
        max_tokens: int
            Maximum tokens in response
        temperature: float
            Sampling temperature

        Returns
        -------
        str
            Generated kernel code
        """
        try:
            response = _call_deepseek_api(
                prompt=prompt, model=self.model, max_tokens=max_tokens, temperature=temperature
            )
            self.last_error = None
            return response
        except Exception as e:
            self.last_error = str(e)
            raise


# Update the main generator to include DeepSeek support
def update_generator_with_deepseek():
    """
    Patch the main AICodegen class to support DeepSeek.
    This function should be called to enable DeepSeek integration.
    """
    from .generator import AICodegen

    original_call_openai = AICodegen._call_openai

    def _call_deepseek(self, prompt: str, max_tokens: int = 1200, temperature: float = 0.0):
        """Call DeepSeek API for code generation."""
        try:
            provider = DeepSeekProvider()
            return provider.generate(prompt, max_tokens, temperature)
        except Exception as e:
            self.last_error = f"DeepSeek request failed: {str(e)}"
            return None

    def _call_provider(self, prompt: str, max_tokens: int = 1200, temperature: float = 0.0, provider: str = "openai"):
        """Call the specified provider."""
        if provider == "deepseek":
            return self._call_deepseek(prompt, max_tokens, temperature)
        else:
            return original_call_openai(self, prompt, max_tokens, temperature)

    # Add the new methods to AICodegen
    AICodegen._call_deepseek = _call_deepseek
    AICodegen._call_provider = _call_provider

    # Update the generate method to support provider selection
    def generate_with_provider(
        self,
        operation_name: str,
        target: Optional[str] = None,
        prompt_extra: Optional[str] = None,
        *,
        temperature: float = 0.0,
        suffix: Optional[str] = None,
        provider: str = "openai",
    ):
        """
        Generate a kernel with specified provider.

        Parameters
        ----------
        provider: str
            AI provider to use ("openai" or "deepseek")
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

        # Try specified provider
        self.last_prompt = prompt
        text = self._call_provider(prompt, temperature=temperature, provider=provider)

        if not text:
            # Surface more helpful diagnostics
            detail = self.last_error or "unknown error"
            raise RuntimeError(
                f"AI code generation failed with provider '{provider}'. "
                f"Model='{self.model}'. Detail: {detail}. "
                "Ensure API key is valid and the model name is correct."
            )

        from .generator import (
            GENERATED_DIR,
            _extract_code_blocks,
            _verify_syntax_python,
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

    # Replace the original generate method
    AICodegen.generate = generate_with_provider


# Initialize DeepSeek integration when module is imported
update_generator_with_deepseek()

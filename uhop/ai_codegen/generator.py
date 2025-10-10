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
    import openai
except Exception:
    openai = None

GENERATED_DIR = Path(__file__).resolve().parent.parent / "generated_kernels"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

def _extract_code_blocks(text: str):
    blocks = re.findall(r"```(?:cuda|c\+\+|c|cpp|python|)\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
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

    def _call_openai(self, prompt: str, max_tokens: int = 1200):
        if openai is None or "OPENAI_API_KEY" not in os.environ:
            return None
        try:
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role":"system","content":"You are an assistant that outputs runnable kernels (CUDA C, OpenCL C, or Python)."},
                    {"role":"user","content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.0
            )
            text = resp["choices"][0]["message"]["content"]
            return text
        except Exception:
            return None

    def _call_deepseek(self, prompt: str):
        if deepseek_provider is None or not getattr(deepseek_provider, "is_configured", lambda: False)():
            return None
        try:
            return deepseek_provider.generate(prompt)
        except Exception:
            return None

    def generate(self, operation_name: str, target: Optional[str] = None, prompt_extra: Optional[str] = None) -> Path:
        """
        Generate a kernel for operation_name.
        target: "cuda", "opencl", "python", "triton" (advisory; generator prefers CUDA).
        Returns path to saved code file in generated_kernels/.
        """
        # Build a flexible prompt depending on target
        if target == "opencl":
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
        text = self._call_openai(prompt)
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
        filename = GENERATED_DIR / f"ai_{operation_name}{ext}"
        # If python target, verify syntax
        if ext == ".py":
            if "import numpy" not in code and "np." in code:
                code = "import numpy as np\n\n" + code
            if not _verify_syntax_python(code):
                raise RuntimeError("AI-generated Python code failed syntax check.")
        # write file
        filename.write_text(code)
        return filename

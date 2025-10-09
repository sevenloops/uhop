# uhop/ai_codegen/generator.py
"""
AI code generator (OpenAI primary). Produces CUDA C kernels as text,
extracts code block, and writes into a .cu file in generated_kernels/.
"""
import os
import re
from pathlib import Path
from typing import Optional

try:
    import openai
except Exception:
    openai = None

GENERATED_DIR = Path(__file__).resolve().parent.parent / "generated_kernels"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

def _extract_code(text: str) -> Optional[str]:
    blocks = re.findall(r"```(?:cuda|c\+\+|c|cpp|)\n(.*?)```", text, flags=re.DOTALL|re.IGNORECASE)
    if blocks:
        return max(blocks, key=len).strip()
    # fallback: if contains 'extern "C"' return entire text
    if 'extern "C"' in text:
        return text.strip()
    return None

class AICodegen:
    def __init__(self, model: str = os.environ.get("UHOP_OPENAI_MODEL", "gpt-4o-mini")):
        self.model = model

    def generate(self, prompt: str, out_name: str = "gen_kernel.cu") -> Path:
        if openai is None or "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError("OpenAI SDK or API key unavailable.")
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role":"system","content":"You are a CUDA kernel author."},
                      {"role":"user","content":prompt}],
            max_tokens=1200,
            temperature=0.0
        )
        text = resp["choices"][0]["message"]["content"]
        code = _extract_code(text)
        if not code:
            raise RuntimeError("No code block found in AI response")
        out_path = GENERATED_DIR / out_name
        out_path.write_text(code)
        return out_path

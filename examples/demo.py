"""Minimal demo for UHOP matmul optimization."""

import os

import numpy as np

from uhop import optimize
from uhop.kernels.numpy_kernels import numpy_matmul


def _load_dotenv_fallback() -> bool:
    # Minimal .env loader: supports KEY=VALUE lines
    candidates = [
        os.getcwd(),
        os.path.dirname(os.path.dirname(__file__)),  # repo root
    ]
    found = False
    for base in candidates:
        path = os.path.join(base, ".env")
        try:
            if os.path.exists(path):
                for line in open(path).read().splitlines():
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if "=" in s:
                        k, v = s.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and v and k not in os.environ:
                            os.environ[k] = v
                found = True
        except Exception:
            continue
    return found


def _ensure_api_key_warning():
    _load_dotenv_fallback()
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY not set. " "AI generation will fail if attempted.")


@optimize("matmul")
def matrix_multiply(a, b):
    return numpy_matmul(a, b)


def main():
    _ensure_api_key_warning()
    print("[UHOP] Starting demo...")
    size = 256
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)

    print("[UHOP] Running matrix_multiply (decorated)...")
    res = matrix_multiply(a, b)
    print(f"Result shape: {res.shape}")
    print("[UHOP] Demo finished.")


if __name__ == "__main__":
    main()

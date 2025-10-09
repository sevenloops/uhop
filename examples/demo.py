# examples/demo.py
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Make sure OPENAI_API_KEY is set in environment
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY not set. AI generation will fail if attempted.")

from uhop import optimize
from uhop.kernels.numpy_kernels import numpy_matmul

# Decorate the baseline implementation
@optimize("matrix_multiply")
def matrix_multiply(a, b):
    return numpy_matmul(a, b)

def main():
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

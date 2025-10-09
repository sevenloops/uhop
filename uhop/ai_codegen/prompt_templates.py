# uhop/ai_codegen/prompt_templates.py
MATMUL_CUDA_PROMPT = """
You are an expert CUDA kernel engineer. Produce a single CUDA C function named
`matmul_kernel` compatible with PyCUDA SourceModule that multiplies two float
matrices A (N x M) and B (M x K), writing to C (N x K). The kernel signature:
extern "C" __global__ void matmul_kernel(const float* A, const float* B, float* C, int N, int M, int K)
Use shared memory tiling and avoid using external libraries. Output only code in a triple-backtick block.
"""

RELU_CUDA_PROMPT = """
Produce a CUDA kernel named `relu_kernel` that applies ReLU in-place on a float array.
Signature: extern "C" __global__ void relu_kernel(float* X, int N)
"""

CONV2D_CUDA_PROMPT = """
Produce a CUDA kernel named `conv2d_kernel` for naive 2D convolution with signature:
extern "C" __global__ void conv2d_kernel(const float* input, const float* kernel, float* output,
                   int H, int W, int KH, int KW, int outH, int outW)
Prefer clarity and correctness first. Output only code in triple-backticks.
"""

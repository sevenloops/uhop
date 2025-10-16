"""
uhop/ai_codegen/prompt_templates.py
-----------------------------------
Prompt library for UHOP AI Kernel Generator.

Each template is crafted to produce deterministic, compilable, and high-performance
kernels across multiple GPU compute APIs (CUDA, HIP, OpenCL, Triton, Metal).

The generator substitutes variables like {operation}, {arch}, or {block} dynamically.
Prompts emphasize:
  - Correct kernel signatures (so auto-compilers work)
  - Proper memory access patterns
  - Synchronization
  - Tiling or local memory where beneficial
  - Fused operation support (e.g., Conv2D + ReLU)
"""

# ======================================================================
# CUDA PROMPTS
# ======================================================================

MATMUL_CUDA_PROMPT = """
You are a senior CUDA compiler engineer. Generate a single CUDA C kernel named
`matmul_kernel` that multiplies two matrices A (N×M) and B (M×K) into C (N×K).

Requirements:
- Signature:
  extern "C" __global__ void matmul_kernel(const float* A, const float* B, float* C, int N, int M, int K)
- Use shared memory tiling to minimize global memory reads.
- Support arbitrary N, M, K values.
- Use 2D thread blocks, each computing a TILE_WIDTH×TILE_WIDTH output tile.
- Ensure coalesced reads and writes.
- Include bounds checks for non-multiple tile sizes.
- Avoid third-party libraries or thrust.
- End with synchronization to ensure correctness.

Return only compilable CUDA code in a triple-backtick block.
"""

RELU_CUDA_PROMPT = """
Generate a CUDA C kernel named `relu_kernel` that performs an in-place ReLU activation
on a 1D float array of length N.

Requirements:
- Signature:
  extern "C" __global__ void relu_kernel(float* X, int N)
- Each thread handles one element.
- Use grid-stride loops for large arrays.
- Use branchless ReLU (e.g., `X[i] = fmaxf(0.0f, X[i])`).
- Output must be syntactically correct CUDA code inside triple backticks.
"""

CONV2D_CUDA_PROMPT = """
You are a CUDA expert generating a naive but correct Conv2D kernel.
Produce a CUDA C kernel named `conv2d_kernel` that performs 2D convolution.

Requirements:
- Signature:
  extern "C" __global__ void conv2d_kernel(
      const float* input, const float* kernel, float* output,
      int H, int W, int KH, int KW, int outH, int outW)
- Implement a direct convolution loop.
- Each thread computes one output pixel.
- Include bounds checks.
- Focus on correctness and clarity; performance optimization is secondary.
- Return only code inside triple backticks.
"""

# ======================================================================
# OPENCL PROMPTS
# ======================================================================

FUSED_OPENCL_CONV2D_RELU = """
You are an OpenCL kernel engineer. Produce a single fused Conv2D+ReLU kernel
named `generated_conv2d_relu`.

Requirements:
- Signature:
  __kernel void generated_conv2d_relu(
      __global const float* input,   // [C_in,H_in,W_in]
      __global const float* weight,  // [C_out,C_in,KH,KW]
      __global float* output,        // [C_out,H_out,W_out]
      const int C_in,const int H_in,const int W_in,
      const int C_out,const int KH,const int KW,
      const int H_out,const int W_out)
- Implement direct convolution with accumulation.
- Fuse ReLU at the end (apply max(0,val)).
- Use local size (8,8,1) and ensure global IDs are bounds-checked.
- Focus on AMD and Intel OpenCL 1.2+ compatibility.
- Output only OpenCL C code in a triple-backtick block.
"""

FUSED_OPENCL_MATMUL_RELU = """
You are an OpenCL compute engineer. Generate a kernel that performs matrix
multiplication followed by ReLU in a single fused pass.

Requirements:
- Kernel name: generated_matmul_relu
- Signature:
  __kernel void generated_matmul_relu(
      const int M,const int N,const int K,
      __global const float* A, __global const float* B, __global float* C);
- Each work item computes one output element (C[i,j]).
- Accumulate over K.
- Apply ReLU (max(0,val)) before writing to C.
- Include bounds checks.
- Optional: small local memory tile for A and B.
- Output only compilable OpenCL C code within triple backticks.
"""

# ======================================================================
# HIP (ROCm) PROMPTS
# ======================================================================

HIP_MATMUL_PROMPT = """
You are an AMD ROCm and HIP kernel engineer.
Produce a HIP C++ kernel equivalent to CUDA matmul.

Requirements:
- Function name: matmul_hip
- Signature:
  extern "C" __global__ void matmul_hip(const float* A, const float* B, float* C, int N, int M, int K)
- Use hipBlockIdx_x/y, hipThreadIdx_x/y appropriately.
- Implement shared-memory tiling identical to CUDA version.
- Support variable matrix dimensions.
- Code must compile under hipcc.
- Output only valid HIP C++ code in triple backticks.
"""

HIP_RELU_PROMPT = """
Generate a HIP kernel named relu_hip that performs in-place ReLU.

Signature:
extern "C" __global__ void relu_hip(float* X, int N)

Each thread updates one element using grid-stride loop.
Use fmaxf(0.0f, X[i]) for branchless ReLU.
Return only compilable HIP C++ code in triple backticks.
"""

HIP_CONV2D_RELU_PROMPT = """
You are a HIP kernel engineer. Produce a fused Conv2D+ReLU kernel for ROCm.

Requirements:
- Kernel name: conv2d_relu_hip
- Signature:
  extern "C" __global__ void conv2d_relu_hip(
      const float* __restrict__ input,
      const float* __restrict__ weight,
      float* __restrict__ output,
      int N,int C,int H,int W,int K,int R,int S,int stride,int pad,int outH,int outW)
- NCHW input, KCRS weights, output N K outH outW. Apply ReLU before store.
- Use shared memory tiling where reasonable. Include bounds checks.
- Output only compilable HIP C++ code within triple backticks.
"""

# ======================================================================
# TRITON PROMPTS
# ======================================================================

TRITON_MATMUL_PROMPT = """
You are generating a Triton kernel in Python for matrix multiplication.

Requirements:
- Kernel function name: matmul_triton
- Accept tensors A, B, C as pointers with strides.
- Use @triton.jit decorator.
- Implement block-wise matmul (BLOCK_M, BLOCK_N, BLOCK_K = 32).
- Use triton.load/store and ensure proper mask for out-of-bounds.
- Compute C = relu(A @ B) if fused variant is requested.
- Return only valid Python Triton kernel code inside triple backticks.
"""

TRITON_CONV2D_RELU_PROMPT = """
Write a Triton kernel that computes fused Conv2D+ReLU on NCHW tensors.

Requirements:
- Decorate with @triton.jit
- Function name: conv2d_relu_triton
- Use program_id for block mapping and masks for OOB.
- Assume small 3x3 kernel and stride/pad provided. Apply ReLU before writing.
- Output valid Triton Python code inside triple backticks.
"""

# ======================================================================
# METAL (Apple M-Series) PROMPTS
# ======================================================================

METAL_MATMUL_PROMPT = """
You are a GPU shader engineer writing in Metal Shading Language (MSL).
Generate a compute kernel for matrix multiplication.

Requirements:
- Kernel name: matmul_msl
- Signature:
  kernel void matmul_msl(
      device const float* A [[buffer(0)]],
      device const float* B [[buffer(1)]],
      device float* C [[buffer(2)]],
      constant int& N [[buffer(3)]],
      constant int& M [[buffer(4)]],
      constant int& K [[buffer(5)]],
      uint2 gid [[thread_position_in_grid]])
- Each thread computes one output element.
- Include bounds check on gid.
- Focus on M2/M3 GPU compatibility (Metal 3.0).
- Output only valid MSL code within triple backticks.
"""

# ======================================================================
# GENERAL AI PROMPT MAP
# ======================================================================

PROMPTS = {
    "cuda": (
        "You are a CUDA kernel engineer. Generate a {operation} kernel optimized for {arch}. "
        "Follow best practices in memory coalescing, shared memory tiling, and warp efficiency. "
        "Return compilable CUDA code only."
    ),
    "opencl": (
        "You are an OpenCL compute expert. Generate an OpenCL C kernel for {operation} "
        "optimized for cross-vendor GPUs (AMD/Intel). Use local memory and barrier synchronization. "
        "Output valid kernel code only."
    ),
    "opencl_fused": (
        "You are an OpenCL performance engineer. Generate a fused OpenCL kernel that merges "
        "{operation} and ReLU activation into one pass to minimize memory bandwidth. "
        "Use explicit bounds-checking and local work-groups for optimal performance."
    ),
    "hip": (
        "You are an AMD ROCm kernel engineer. Generate a HIP kernel equivalent to the CUDA version "
        "for {operation}. Ensure correct block and grid indexing and HIP-specific syntax."
    ),
    "triton": (
        "You are a Triton kernel author. Write a @triton.jit kernel for {operation} "
        "optimized for block size {block}, using load/store masking and program_id-based indexing."
    ),
    "metal": (
        "You are an Apple Metal shader developer. Generate a Metal compute kernel for {operation}. "
        "Ensure compatibility with Apple M-series GPUs and MSL 3.0 standards."
    ),
    "vulkan": (
        "You are a Vulkan compute shader expert. Generate a GLSL/SPIR-V compute shader implementing {operation}. "
        "Use shared memory if needed and ensure SPIR-V compatibility for all major GPU vendors."
    ),
}

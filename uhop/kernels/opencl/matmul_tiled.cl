// uhop/kernels/opencl/matmul_tiled.cl
// Tiled matrix multiplication using local memory (1D local arrays for portability).
// C[N,K] = A[N,M] x B[M,K]
// Global size: (K, N), Local size: (TILE, TILE)

#ifndef TILE
#define TILE 16
#endif

// Optional vector width hint (may be used by future optimized variants)
#ifndef VEC
#define VEC 1
#endif

__kernel void matmul_tiled(__global const float *A, __global const float *B,
                           __global float *C, const int N, const int M,
                           const int K) {
#ifdef GWS_FLIP
  const int row = get_global_id(0);
  const int col = get_global_id(1);
#else
  const int row = get_global_id(1);
  const int col = get_global_id(0);
#endif

  // Use 1D local arrays to avoid potential compiler quirks with 2D arrays
  __local float As[TILE * TILE];
  __local float Bs[TILE * TILE];

  float acc = 0.0f;
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);

  const int tiles = (M + TILE - 1) / TILE;

  for (int t = 0; t < tiles; ++t) {
    int a_col = t * TILE + lx;
    int b_row = t * TILE + ly;

    // Load tiles with bounds checks
    As[ly * TILE + lx] = (row < N && a_col < M) ? A[row * M + a_col] : 0.0f;
    Bs[ly * TILE + lx] = (b_row < M && col < K) ? B[b_row * K + col] : 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute partial product
    for (int k2 = 0; k2 < TILE; ++k2) {
      float a_val = As[ly * TILE + k2];
      float b_val = Bs[k2 * TILE + lx];
      acc += a_val * b_val;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (row < N && col < K) {
    C[row * K + col] = acc;
  }
}

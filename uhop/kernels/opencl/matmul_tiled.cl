// uhop/kernels/opencl/matmul_tiled.cl
// Tiled matrix multiplication using local memory.
// C[N,K] = A[N,M] x B[M,K]
// Global size: (K, N), Local size: (TILE, TILE)

#ifndef TILE
#define TILE 16
#endif

// Optional vector width hint (may be used by future optimized variants)
#ifndef VEC
#define VEC 1
#endif

__kernel void matmul_tiled(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int N, const int M, const int K)
{
    const int row = get_global_id(1);
    const int col = get_global_id(0);

    __local float As[TILE][TILE];
    __local float Bs[TILE][TILE];

    float acc = 0.0f;
    const int local_row = get_local_id(1);
    const int local_col = get_local_id(0);

    const int tiles = (M + TILE - 1) / TILE;

    for (int t = 0; t < tiles; ++t) {
        int a_col = t * TILE + local_col;
        int b_row = t * TILE + local_row;

        // Load tiles with bounds checks
        if (row < N && a_col < M) {
            As[local_row][local_col] = A[row * M + a_col];
        } else {
            As[local_row][local_col] = 0.0f;
        }

        if (b_row < M && col < K) {
            Bs[local_row][local_col] = B[b_row * K + col];
        } else {
            Bs[local_row][local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial product
        for (int k2 = 0; k2 < TILE; ++k2) {
            acc += As[local_row][k2] * Bs[k2][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < N && col < K) {
        C[row * K + col] = acc;
    }
}

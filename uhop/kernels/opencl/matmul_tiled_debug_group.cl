// Debug variant for arbitrary workgroup tile: emits per-k-tile partial accumulators
// for the tile at group coords (GX, GY). Computes row/col from GX,GY and local ids,
// avoiding use of get_global_id to isolate group-mapping issues.

#ifndef TILE
#define TILE 16
#endif

__kernel void matmul_tiled_debug_group(
    __global const float *A, __global const float *B,
    __global float *Cdbg, const int N, const int M, const int K,
    const int GX, const int GY) {
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int row = GY * TILE + ly;
  const int col = GX * TILE + lx;

  __local float As[TILE * TILE];
  __local float Bs[TILE * TILE];

  const int tiles = (M + TILE - 1) / TILE;
  for (int t = 0; t < tiles; ++t) {
    int a_col = t * TILE + lx;
    int b_row = t * TILE + ly;
    As[ly * TILE + lx] = (row < N && a_col < M) ? A[row * M + a_col] : 0.0f;
    Bs[ly * TILE + lx] = (b_row < M && col < K) ? B[b_row * K + col] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    float tile_acc = 0.0f;
    for (int k2 = 0; k2 < TILE; ++k2) {
      tile_acc += As[ly * TILE + k2] * Bs[k2 * TILE + lx];
    }
    Cdbg[t * TILE * TILE + ly * TILE + lx] = tile_acc;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

// Debug variant: emits per-k-tile partial accumulators for the first workgroup (0,0)
// Cdbg stores for each t in [0..tiles): TILE x TILE partial contributions to C
// Layout: Cdbg[t * TILE * TILE + ly * TILE + lx] for ly,lx in [0..TILE)

#ifndef TILE
#define TILE 16
#endif

__kernel void matmul_tiled_debug(__global const float *A, __global const float *B,
                                 __global float *Cdbg, const int N, const int M,
                                 const int K) {
  
#ifdef GWS_FLIP
  const int row = get_global_id(0);
  const int col = get_global_id(1);
#else
  const int row = get_global_id(1);
  const int col = get_global_id(0);
#endif
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);

  __local float As[TILE * TILE];
  __local float Bs[TILE * TILE];

  const int tiles = (M + TILE - 1) / TILE;
  float acc_total = 0.0f;
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
    acc_total += tile_acc;

    // Only first workgroup dumps debug data
    if (get_group_id(0) == 0 && get_group_id(1) == 0 && ly < TILE && lx < TILE) {
      Cdbg[t * TILE * TILE + ly * TILE + lx] = tile_acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

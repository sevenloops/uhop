// Dump loaded As and Bs tiles for a given tile index t for the first workgroup
#ifndef TILE
#define TILE 16
#endif

__kernel void matmul_tiled_dump_loads(__global const float *A, __global const float *B,
                                      __global float *As_out, __global float *Bs_out,
                                      const int N, const int M, const int K,
                                      const int TIDX) {
  
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

  int a_col = TIDX * TILE + lx;
  int b_row = TIDX * TILE + ly;
  As[ly * TILE + lx] = (row < N && a_col < M) ? A[row * M + a_col] : 0.0f;
  Bs[ly * TILE + lx] = (b_row < M && col < K) ? B[b_row * K + col] : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (get_group_id(0) == 0 && get_group_id(1) == 0 && ly < TILE && lx < TILE) {
    As_out[ly * TILE + lx] = As[ly * TILE + lx];
    Bs_out[ly * TILE + lx] = Bs[ly * TILE + lx];
  }
}

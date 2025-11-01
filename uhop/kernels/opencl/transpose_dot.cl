// uhop/kernels/opencl/transpose_dot.cl

__kernel void transpose2d(__global const float *X, __global float *Y,
                          const int rows, const int cols) {
  int r = get_global_id(1);
  int c = get_global_id(0);
  if (r < rows && c < cols) {
    Y[c * rows + r] = X[r * cols + c];
  }
}

__kernel void dot_product(__global const float *A, __global const float *B,
                          __global float *Out, const int N) {
  __local float sdata[256];
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int gsize = get_global_size(0);
  int lsize = get_local_size(0);

  float sum = 0.0f;
  for (int i = gid; i < N; i += gsize)
    sum += A[i] * B[i];
  sdata[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = lsize / 2; s > 0; s >>= 1) {
    if (lid < s)
      sdata[lid] += sdata[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0)
    Out[get_group_id(0)] = sdata[0];
}

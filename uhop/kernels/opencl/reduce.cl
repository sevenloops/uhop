// uhop/kernels/opencl/reduce.cl
// Reduction kernels for OpenCL (two-stage: partials then finalize)

#ifndef FLT_MAX
#define FLT_MAX 3.402823e38f
#endif

// Sum reduction - compute partial sums per workgroup
__kernel void reduce_sum_partials(__global const float *X,
                                  __global float *partials, const int N) {
  __local float sdata[256];
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int gsize = get_global_size(0);
  int lsize = get_local_size(0);

  float sum = 0.0f;
  for (int i = gid; i < N; i += gsize)
    sum += X[i];
  sdata[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = lsize / 2; s > 0; s >>= 1) {
    if (lid < s)
      sdata[lid] += sdata[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0)
    partials[get_group_id(0)] = sdata[0];
}

__kernel void reduce_mean_finalize(__global const float *partials,
                                   __global float *Out, const int num_partials,
                                   const int N) {
  float sum = 0.0f;
  for (int i = 0; i < num_partials; ++i)
    sum += partials[i];
  Out[0] = sum / (float)N;
}

__kernel void reduce_sum_finalize(__global const float *partials,
                                  __global float *Out, const int num_partials) {
  float sum = 0.0f;
  for (int i = 0; i < num_partials; ++i)
    sum += partials[i];
  Out[0] = sum;
}

// Max/min reduction using two-stage approach
__kernel void reduce_max_partials(__global const float *X,
                                  __global float *partials, const int N) {
  __local float sdata[256];
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int gsize = get_global_size(0);
  int lsize = get_local_size(0);
  float vmax = -FLT_MAX;
  for (int i = gid; i < N; i += gsize)
    vmax = fmax(vmax, X[i]);
  sdata[lid] = vmax;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = lsize / 2; s > 0; s >>= 1) {
    if (lid < s)
      sdata[lid] = fmax(sdata[lid], sdata[lid + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0)
    partials[get_group_id(0)] = sdata[0];
}

__kernel void reduce_min_partials(__global const float *X,
                                  __global float *partials, const int N) {
  __local float sdata[256];
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int gsize = get_global_size(0);
  int lsize = get_local_size(0);
  float vmin = FLT_MAX;
  for (int i = gid; i < N; i += gsize)
    vmin = fmin(vmin, X[i]);
  sdata[lid] = vmin;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = lsize / 2; s > 0; s >>= 1) {
    if (lid < s)
      sdata[lid] = fmin(sdata[lid], sdata[lid + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0)
    partials[get_group_id(0)] = sdata[0];
}

__kernel void reduce_max_finalize(__global const float *partials,
                                  __global float *Out, const int num_partials) {
  float vmax = -FLT_MAX;
  for (int i = 0; i < num_partials; ++i)
    vmax = fmax(vmax, partials[i]);
  Out[0] = vmax;
}

__kernel void reduce_min_finalize(__global const float *partials,
                                  __global float *Out, const int num_partials) {
  float vmin = FLT_MAX;
  for (int i = 0; i < num_partials; ++i)
    vmin = fmin(vmin, partials[i]);
  Out[0] = vmin;
}

// L2 norm: sqrt(sum(x^2))
__kernel void reduce_norm_partials(__global const float *X,
                                   __global float *partials, const int N) {
  __local float sdata[256];
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int gsize = get_global_size(0);
  int lsize = get_local_size(0);
  float sum = 0.0f;
  for (int i = gid; i < N; i += gsize)
    sum += X[i] * X[i];
  sdata[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = lsize / 2; s > 0; s >>= 1) {
    if (lid < s)
      sdata[lid] += sdata[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0)
    partials[get_group_id(0)] = sdata[0];
}

__kernel void reduce_norm_finalize(__global const float *partials,
                                   __global float *Out,
                                   const int num_partials) {
  float sum = 0.0f;
  for (int i = 0; i < num_partials; ++i)
    sum += partials[i];
  Out[0] = sqrt(sum);
}

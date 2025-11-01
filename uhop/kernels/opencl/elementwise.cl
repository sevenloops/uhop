// uhop/kernels/opencl/elementwise.cl
// Common elementwise operations for OpenCL

__kernel void elementwise_add(__global const float *A, __global const float *B,
                              __global float *Out, const int N) {
  int i = get_global_id(0);
  if (i < N) {
    Out[i] = A[i] + B[i];
  }
}

__kernel void elementwise_sub(__global const float *A, __global const float *B,
                              __global float *Out, const int N) {
  int i = get_global_id(0);
  if (i < N) {
    Out[i] = A[i] - B[i];
  }
}

__kernel void elementwise_mul(__global const float *A, __global const float *B,
                              __global float *Out, const int N) {
  int i = get_global_id(0);
  if (i < N) {
    Out[i] = A[i] * B[i];
  }
}

__kernel void elementwise_div(__global const float *A, __global const float *B,
                              __global float *Out, const int N,
                              const float eps) {
  int i = get_global_id(0);
  if (i < N) {
    float denom = B[i];
    if (fabs(denom) < eps)
      denom = (denom >= 0.0f ? eps : -eps);
    Out[i] = A[i] / denom;
  }
}

__kernel void elementwise_pow(__global const float *A, __global const float *B,
                              __global float *Out, const int N) {
  int i = get_global_id(0);
  if (i < N) {
    Out[i] = pow(A[i], B[i]);
  }
}

__kernel void elementwise_max(__global const float *A, __global const float *B,
                              __global float *Out, const int N) {
  int i = get_global_id(0);
  if (i < N) {
    Out[i] = fmax(A[i], B[i]);
  }
}

__kernel void elementwise_min(__global const float *A, __global const float *B,
                              __global float *Out, const int N) {
  int i = get_global_id(0);
  if (i < N) {
    Out[i] = fmin(A[i], B[i]);
  }
}

__kernel void elementwise_leakyrelu(__global const float *X,
                                    __global float *Out, const int N,
                                    const float alpha) {
  int i = get_global_id(0);
  if (i < N) {
    float v = X[i];
    Out[i] = (v > 0.0f) ? v : (alpha * v);
  }
}

__kernel void elementwise_exp(__global const float *X, __global float *Out,
                              const int N) {
  int i = get_global_id(0);
  if (i < N)
    Out[i] = exp(X[i]);
}

__kernel void elementwise_log(__global const float *X, __global float *Out,
                              const int N, const float eps) {
  int i = get_global_id(0);
  if (i < N)
    Out[i] = log(fmax(fabs(X[i]), eps));
}

__kernel void elementwise_sqrt(__global const float *X, __global float *Out,
                               const int N) {
  int i = get_global_id(0);
  if (i < N)
    Out[i] = sqrt(fabs(X[i]));
}

__kernel void sigmoid_kernel(__global float *X, const int N) {
  int i = get_global_id(0);
  if (i < N) {
    float x = X[i];
    X[i] = 1.0f / (1.0f + exp(-x));
  }
}

__kernel void tanh_kernel(__global float *X, const int N) {
  int i = get_global_id(0);
  if (i < N) {
    float x = X[i];
    float epos = exp(x);
    float eneg = exp(-x);
    X[i] = (epos - eneg) / (epos + eneg);
  }
}

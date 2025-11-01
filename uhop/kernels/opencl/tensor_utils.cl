// uhop/kernels/opencl/tensor_utils.cl

// Reshape is a no-op on raw buffers; provided for completeness
__kernel void reshape_noop(__global const float *X, __global float *Y,
                           const int N) {
  int i = get_global_id(0);
  if (i < N)
    Y[i] = X[i];
}

// Slice 2D: copy [h0:h1, w0:w1] from src (H,W) to dst (h1-h0, w1-w0)
__kernel void slice2d(__global const float *X, __global float *Y, const int H,
                      const int W, const int h0, const int h1, const int w0,
                      const int w1) {
  int ow = get_global_id(0);
  int oh = get_global_id(1);
  int outH = h1 - h0;
  int outW = w1 - w0;
  if (oh < outH && ow < outW) {
    int ih = h0 + oh;
    int iw = w0 + ow;
    Y[oh * outW + ow] = X[ih * W + iw];
  }
}

// Concat 1D: concatenate A and B into Out
__kernel void concat1d(__global const float *A, __global const float *B,
                       __global float *Out, const int NA, const int NB) {
  int i = get_global_id(0);
  int N = NA + NB;
  if (i < N) {
    Out[i] = (i < NA) ? A[i] : B[i - NA];
  }
}

// Pad 2D constant
__kernel void pad2d_constant(__global const float *X, __global float *Y,
                             const int H, const int W, const int pad_top,
                             const int pad_left, const int outH, const int outW,
                             const float pad_val) {
  int ow = get_global_id(0);
  int oh = get_global_id(1);
  if (oh >= outH || ow >= outW)
    return;
  int ih = oh - pad_top;
  int iw = ow - pad_left;
  float v = pad_val;
  if (ih >= 0 && ih < H && iw >= 0 && iw < W)
    v = X[ih * W + iw];
  Y[oh * outW + ow] = v;
}

// Broadcast to 2D from 1D: Out[H,W] from X[W] (row broadcast)
__kernel void broadcast_row(__global const float *X, __global float *Y,
                            const int H, const int W) {
  int ow = get_global_id(0);
  int oh = get_global_id(1);
  if (oh < H && ow < W) {
    Y[oh * W + ow] = X[ow];
  }
}

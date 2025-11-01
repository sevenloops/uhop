// uhop/kernels/opencl/conv2d.cl
// Valid 3D NDRange Conv2D kernel (NCHW) with stride and padding.
// Global IDs: g0 = outX, g1 = outY, g2 = (b * K + k)

__kernel void
conv2d(__global const float *input,   // [B,C,H, W] in row-major flattened
       __global const float *weights, // [K,C,R,S]
       __global float *output,        // [B,K,outH,outW]
       const int B, const int C, const int H, const int W, const int K,
       const int R, const int S, const int outH, const int outW,
       const int stride, const int pad) {
  int out_x = get_global_id(0);
  int out_y = get_global_id(1);
  int z = get_global_id(2);
  if (out_x >= outW || out_y >= outH)
    return;
  int k = z % K;
  int b = z / K;
  if (b >= B)
    return;

  float acc = 0.0f;
  // Accumulate over input channels and kernel window
  for (int c = 0; c < C; ++c) {
    for (int r = 0; r < R; ++r) {
      int in_y = out_y * stride + r - pad;
      if (in_y < 0 || in_y >= H)
        continue;
      for (int s2 = 0; s2 < S; ++s2) {
        int in_x = out_x * stride + s2 - pad;
        if (in_x < 0 || in_x >= W)
          continue;
        int in_idx = ((b * C + c) * H + in_y) * W + in_x;
        int w_idx = ((k * C + c) * R + r) * S + s2;
        acc += input[in_idx] * weights[w_idx];
      }
    }
  }
  int out_idx = ((b * K + k) * outH + out_y) * outW + out_x;
  output[out_idx] = acc;
}

// uhop/kernels/opencl/conv2d_relu.cl
// Fused Conv2D + ReLU for OpenCL
// Supports NCHW layout, single batch, float32

#define TILE_WIDTH 8
#define KERNEL_MAX 11 // max supported kernel size

__kernel void
conv2d_relu(__global const float *input,   // [C_in, H_in, W_in]
            __global const float *weights, // [C_out, C_in, K_h, K_w]
            __global const float *bias,    // [C_out]
            __global float *output,        // [C_out, H_out, W_out]
            const int C_in, const int H_in, const int W_in, const int C_out,
            const int K_h, const int K_w, const int stride, const int pad_h,
            const int pad_w, const int H_out, const int W_out) {
  // Thread coordinates
  const int out_x = get_global_id(0);
  const int out_y = get_global_id(1);
  const int out_c = get_global_id(2);

  if (out_x >= W_out || out_y >= H_out || out_c >= C_out)
    return;

  float acc = 0.0f;

  // Convolution sum
  for (int c = 0; c < C_in; ++c) {
    for (int ky = 0; ky < K_h; ++ky) {
      for (int kx = 0; kx < K_w; ++kx) {
        int in_y = out_y * stride + ky - pad_h;
        int in_x = out_x * stride + kx - pad_w;
        if (in_y >= 0 && in_y < H_in && in_x >= 0 && in_x < W_in) {
          int in_idx = c * H_in * W_in + in_y * W_in + in_x;
          int w_idx =
              out_c * (C_in * K_h * K_w) + c * (K_h * K_w) + ky * K_w + kx;
          acc += input[in_idx] * weights[w_idx];
        }
      }
    }
  }

  // Add bias and apply ReLU
  acc += bias[out_c];
  if (acc < 0.0f)
    acc = 0.0f;

  // Write to output
  int out_idx = out_c * H_out * W_out + out_y * W_out + out_x;
  output[out_idx] = acc;
}

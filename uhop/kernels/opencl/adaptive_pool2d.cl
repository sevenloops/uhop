// uhop/kernels/opencl/adaptive_pool2d.cl
// Adaptive average and max pooling to a target size (outH, outW)

#ifndef FLT_MAX
#define FLT_MAX 3.402823e38f
#endif

__kernel void adaptive_avgpool2d(__global const float *input,
                                 __global float *output, const int N,
                                 const int C, const int H, const int W,
                                 const int outH, const int outW) {
  int ow = get_global_id(0);
  int oh = get_global_id(1);
  int c = get_global_id(2) % C;
  int n = get_global_id(2) / C;
  if (n >= N || c >= C || oh >= outH || ow >= outW)
    return;

  int hstart = (oh * H) / outH;
  int hend = ((oh + 1) * H + outH - 1) / outH;
  int wstart = (ow * W) / outW;
  int wend = ((ow + 1) * W + outW - 1) / outW;

  float sum = 0.0f;
  int count = 0;
  for (int ih = hstart; ih < hend; ++ih) {
    for (int iw = wstart; iw < wend; ++iw) {
      int idx = ((n * C + c) * H + ih) * W + iw;
      sum += input[idx];
      count++;
    }
  }
  int out_idx = ((n * C + c) * outH + oh) * outW + ow;
  output[out_idx] = sum / (float)(count > 0 ? count : 1);
}

__kernel void adaptive_maxpool2d(__global const float *input,
                                 __global float *output, const int N,
                                 const int C, const int H, const int W,
                                 const int outH, const int outW) {
  int ow = get_global_id(0);
  int oh = get_global_id(1);
  int c = get_global_id(2) % C;
  int n = get_global_id(2) / C;
  if (n >= N || c >= C || oh >= outH || ow >= outW)
    return;

  int hstart = (oh * H) / outH;
  int hend = ((oh + 1) * H + outH - 1) / outH;
  int wstart = (ow * W) / outW;
  int wend = ((ow + 1) * W + outW - 1) / outW;

  float vmax = -FLT_MAX;
  for (int ih = hstart; ih < hend; ++ih) {
    for (int iw = wstart; iw < wend; ++iw) {
      int idx = ((n * C + c) * H + ih) * W + iw;
      vmax = fmax(vmax, input[idx]);
    }
  }
  int out_idx = ((n * C + c) * outH + oh) * outW + ow;
  output[out_idx] = vmax;
}

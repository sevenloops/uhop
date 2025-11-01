// uhop/kernels/opencl/pooling.cl
// Pooling operations for OpenCL

__kernel void maxpool2d_kernel(__global const float *input,
                               __global float *output, int batch_size,
                               int channels, int input_h, int input_w,
                               int kernel_h, int kernel_w, int stride_h,
                               int stride_w, int padding_h, int padding_w) {
  int output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1;
  int output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1;

  int idx = get_global_id(0);
  int total_output = batch_size * channels * output_h * output_w;

  if (idx >= total_output)
    return;

  int ow = idx % output_w;
  int oh = (idx / output_w) % output_h;
  int c = (idx / (output_w * output_h)) % channels;
  int b = idx / (output_w * output_h * channels);

  float max_val = -1e38f;

  for (int kh = 0; kh < kernel_h; ++kh) {
    for (int kw = 0; kw < kernel_w; ++kw) {
      int ih = oh * stride_h + kh - padding_h;
      int iw = ow * stride_w + kw - padding_w;

      if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
        int in_idx = b * channels * input_h * input_w + c * input_h * input_w +
                     ih * input_w + iw;
        max_val = fmax(max_val, input[in_idx]);
      }
    }
  }

  output[idx] = max_val;
}

__kernel void avgpool2d_kernel(__global const float *input,
                               __global float *output, int batch_size,
                               int channels, int input_h, int input_w,
                               int kernel_h, int kernel_w, int stride_h,
                               int stride_w, int padding_h, int padding_w,
                               int count_include_pad) {
  int output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1;
  int output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1;

  int idx = get_global_id(0);
  int total_output = batch_size * channels * output_h * output_w;

  if (idx >= total_output)
    return;

  int ow = idx % output_w;
  int oh = (idx / output_w) % output_h;
  int c = (idx / (output_w * output_h)) % channels;
  int b = idx / (output_w * output_h * channels);

  float sum = 0.0f;
  int count = 0;

  for (int kh = 0; kh < kernel_h; ++kh) {
    for (int kw = 0; kw < kernel_w; ++kw) {
      int ih = oh * stride_h + kh - padding_h;
      int iw = ow * stride_w + kw - padding_w;

      if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
        int in_idx = b * channels * input_h * input_w + c * input_h * input_w +
                     ih * input_w + iw;
        sum += input[in_idx];
        count++;
      } else if (count_include_pad) {
        count++;
      }
    }
  }

  output[idx] = sum / (float)((count > 0) ? count : 1);
}

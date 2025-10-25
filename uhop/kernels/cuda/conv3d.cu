// uhop/kernels/cuda/conv3d.cu
// 3D Convolution kernel

extern "C" __global__
void conv3d_kernel(const float* input, const float* weight, const float* bias,
                   float* output, int batch_size, int in_channels, int out_channels,
                   int input_depth, int input_height, int input_width,
                   int kernel_d, int kernel_h, int kernel_w,
                   int stride_d, int stride_h, int stride_w,
                   int padding_d, int padding_h, int padding_w) {
    int out_depth = (input_depth + 2 * padding_d - kernel_d) / stride_d + 1;
    int out_height = (input_height + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_width = (input_width + 2 * padding_w - kernel_w) / stride_w + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (idx >= total_output) return;
    
    // Decode output indices
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int d = (idx / (out_width * out_height)) % out_depth;
    int oc = (idx / (out_width * out_height * out_depth)) % out_channels;
    int b = idx / (out_width * out_height * out_depth * out_channels);
    
    float sum = bias ? bias[oc] : 0.0f;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int in_d = d * stride_d + kd - padding_d;
                    int in_h = h * stride_h + kh - padding_h;
                    int in_w = w * stride_w + kw - padding_w;
                    
                    if (in_d >= 0 && in_d < input_depth &&
                        in_h >= 0 && in_h < input_height &&
                        in_w >= 0 && in_w < input_width) {
                        int in_idx = b * in_channels * input_depth * input_height * input_width +
                                     ic * input_depth * input_height * input_width +
                                     in_d * input_height * input_width +
                                     in_h * input_width + in_w;
                        int w_idx = oc * in_channels * kernel_d * kernel_h * kernel_w +
                                    ic * kernel_d * kernel_h * kernel_w +
                                    kd * kernel_h * kernel_w + kh * kernel_w + kw;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
    }
    
    output[idx] = sum;
}

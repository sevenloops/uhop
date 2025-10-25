// uhop/kernels/opencl/conv1d.cl
// 1D Convolution for OpenCL

__kernel void conv1d_kernel(__global const float* input,
                            __global const float* weight,
                            __global const float* bias,
                            __global float* output,
                            int batch_size, int in_channels, int out_channels,
                            int input_length, int kernel_size,
                            int stride, int padding) {
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    
    int b = get_global_id(2);
    int oc = get_global_id(1);
    int out_idx = get_global_id(0);
    
    if (b >= batch_size || oc >= out_channels || out_idx >= output_length) return;
    
        // OpenCL C: use 0 pointer check instead of NULL
        float sum = (bias != 0) ? bias[oc] : 0.0f;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int in_idx = out_idx * stride + k - padding;
            if (in_idx >= 0 && in_idx < input_length) {
                float in_val = input[b * in_channels * input_length + ic * input_length + in_idx];
                float w_val = weight[oc * in_channels * kernel_size + ic * kernel_size + k];
                sum += in_val * w_val;
            }
        }
    }
    
    output[b * out_channels * output_length + oc * output_length + out_idx] = sum;
}

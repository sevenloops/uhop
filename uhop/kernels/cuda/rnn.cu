// uhop/kernels/cuda/rnn.cu
// Vanilla RNN cell kernel

extern "C" __global__
void rnn_cell_kernel(const float* input, const float* hidden, 
                     const float* weight_ih, const float* weight_hh,
                     const float* bias_ih, const float* bias_hh,
                     float* output, int batch_size, int input_size, int hidden_size) {
    int b = blockIdx.x;
    int h = threadIdx.x;
    
    if (b >= batch_size || h >= hidden_size) return;
    
    float sum = 0.0f;
    
    // Input to hidden: weight_ih @ input
    for (int i = 0; i < input_size; ++i) {
        sum += weight_ih[h * input_size + i] * input[b * input_size + i];
    }
    if (bias_ih) sum += bias_ih[h];
    
    // Hidden to hidden: weight_hh @ hidden
    for (int i = 0; i < hidden_size; ++i) {
        sum += weight_hh[h * hidden_size + i] * hidden[b * hidden_size + i];
    }
    if (bias_hh) sum += bias_hh[h];
    
    // Apply tanh activation
    output[b * hidden_size + h] = tanhf(sum);
}

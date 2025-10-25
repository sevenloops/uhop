// uhop/kernels/cuda/lstm.cu
// LSTM cell kernel

extern "C" __global__
void lstm_cell_kernel(const float* input, const float* hidden, const float* cell,
                      const float* weight_ih, const float* weight_hh,
                      const float* bias_ih, const float* bias_hh,
                      float* output_hidden, float* output_cell,
                      int batch_size, int input_size, int hidden_size) {
    int b = blockIdx.x;
    int h = threadIdx.x;
    
    if (b >= batch_size || h >= hidden_size) return;
    
    // LSTM has 4 gates: input, forget, cell, output (i, f, g, o)
    // Each needs computation, so we'll compute all 4
    
    float gates[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    for (int gate = 0; gate < 4; ++gate) {
        float sum = 0.0f;
        
        // Input to hidden
        for (int i = 0; i < input_size; ++i) {
            int w_idx = (gate * hidden_size + h) * input_size + i;
            sum += weight_ih[w_idx] * input[b * input_size + i];
        }
        if (bias_ih) sum += bias_ih[gate * hidden_size + h];
        
        // Hidden to hidden
        for (int i = 0; i < hidden_size; ++i) {
            int w_idx = (gate * hidden_size + h) * hidden_size + i;
            sum += weight_hh[w_idx] * hidden[b * hidden_size + i];
        }
        if (bias_hh) sum += bias_hh[gate * hidden_size + h];
        
        gates[gate] = sum;
    }
    
    // Apply activations
    float i_gate = 1.0f / (1.0f + expf(-gates[0]));  // sigmoid
    float f_gate = 1.0f / (1.0f + expf(-gates[1]));  // sigmoid
    float g_gate = tanhf(gates[2]);                   // tanh
    float o_gate = 1.0f / (1.0f + expf(-gates[3]));  // sigmoid
    
    // Update cell state
    float c_new = f_gate * cell[b * hidden_size + h] + i_gate * g_gate;
    output_cell[b * hidden_size + h] = c_new;
    
    // Update hidden state
    output_hidden[b * hidden_size + h] = o_gate * tanhf(c_new);
}

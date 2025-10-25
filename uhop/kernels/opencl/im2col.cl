// uhop/kernels/opencl/im2col.cl
// Convert input image block to column matrix for GEMM-based conv
// Input: X[B,C,H,W]
// Output (for batch b): Col[(C*R*S), (outH*outW)] in row-major
// Global size: (outW, outH, C*R*S)

__kernel void im2col_batched(
    __global const float* input,  // [B,C,H,W]
    __global float* columns,      // [C*R*S, outH*outW]
    const int B,
    const int C,
    const int H,
    const int W,
    const int R,
    const int S,
    const int outH,
    const int outW,
    const int stride,
    const int pad,
    const int batch_index
) {
    int ow = get_global_id(0);
    int oh = get_global_id(1);
    int crs = get_global_id(2); // 0..C*R*S-1

    if (ow >= outW || oh >= outH) return;

    // Decode crs into (c,r,s)
    int c = crs / (R*S);
    int rs = crs % (R*S);
    int r = rs / S;
    int s = rs % S;

    // Compute input coords
    int in_y = oh * stride - pad + r;
    int in_x = ow * stride - pad + s;

    float val = 0.0f;
    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
        int in_idx = (((batch_index * C + c) * H) + in_y) * W + in_x;
        val = input[in_idx];
    }

    // Column index: (crs, oh*outW + ow)
    int col_row = crs;
    int col_col = oh * outW + ow;
    int out_idx = col_row * (outW * outH) + col_col;
    columns[out_idx] = val;
}

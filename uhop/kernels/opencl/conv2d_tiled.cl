// uhop/kernels/opencl/conv2d_tiled.cl
// Tiled Conv2D kernel with local memory for input tiles and weights.
// Layout: NCHW input, KCRS weights. Output N K outH outW.
// Global: (outW, outH, N*K). Local: (TILE_W, TILE_H, 1)
// Assumes small kernels (e.g., 3x3, 5x5). Uses dynamic local memory passed as kernel args.

#define TILE_H 8
#define TILE_W 8

__kernel void conv2d_tiled(
    __global const float* input,   // [B,C,H,W]
    __global const float* weight,  // [K,C,R,S]
    __global float* output,        // [B,K,outH,outW]
    const int B, const int C, const int H, const int W,
    const int K, const int R, const int S,
    const int outH, const int outW, const int stride, const int pad,
    __local float* tile_in,
    __local float* tile_w
) {
    int out_x = get_global_id(0);
    int out_y = get_global_id(1);
    int z = get_global_id(2);
    int co = z % K;
    int b = z / K;
    // Do not early-return on out-of-range XY to keep barriers well-defined.
    // We will guard the final store.
    if (b >= B) return; // z dimension matches exactly N*K, but keep safety.

    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int group_x = get_group_id(0);
    int group_y = get_group_id(1);

    // Tile origin in output space
    int tile_out_x0 = group_x * TILE_W;
    int tile_out_y0 = group_y * TILE_H;

    // Corresponding input tile extent for this output tile considering stride
    // For an output tile of size TILE_H x TILE_W, the input span is
    // (TILE_H - 1) * stride + R by (TILE_W - 1) * stride + S
    int tile_in_w = (TILE_W - 1) * stride + S;
    int tile_in_h = (TILE_H - 1) * stride + R;

    // Base input coordinates (top-left) for this tile before adding local offsets
    int base_in_y = tile_out_y0 * stride - pad;
    int base_in_x = tile_out_x0 * stride - pad;

    float acc = 0.0f;

    for (int ci = 0; ci < C; ++ci) {
        // Cooperative load of input tile for channel ci into local memory
        for (int ty = ly; ty < tile_in_h; ty += get_local_size(1)) {
            for (int tx = lx; tx < tile_in_w; tx += get_local_size(0)) {
                int in_y = base_in_y + ty;
                int in_x = base_in_x + tx;
                float v = 0.0f;
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    int in_idx = ((b * C + ci) * H + in_y) * W + in_x;
                    v = input[in_idx];
                }
                tile_in[ty * tile_in_w + tx] = v;
            }
        }
        // Cooperative load of weights for (co,ci)
        for (int ty = ly; ty < R; ty += get_local_size(1)) {
            for (int tx = lx; tx < S; tx += get_local_size(0)) {
                int w_idx = ((co * C + ci) * R + ty) * S + tx;
                tile_w[ty * S + tx] = weight[w_idx];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute this work-item's output using local tiles
        int ox = out_x - tile_out_x0;
        int oy = out_y - tile_out_y0;
        if (ox >= 0 && ox < TILE_W && oy >= 0 && oy < TILE_H) {
            float sum = 0.0f;
            for (int r = 0; r < R; ++r) {
                for (int s2 = 0; s2 < S; ++s2) {
                    int ti_y = oy * stride + r;
                    int ti_x = ox * stride + s2;
                    sum += tile_in[ti_y * tile_in_w + ti_x] * tile_w[r * S + s2];
                }
            }
            acc += sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (out_x < outW && out_y < outH) {
        int out_idx = ((b * K + co) * outH + out_y) * outW + out_x;
        output[out_idx] = acc;
    }
}

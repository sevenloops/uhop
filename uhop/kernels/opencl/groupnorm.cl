// uhop/kernels/opencl/groupnorm.cl
// Group Normalization for NCHW layout

__kernel void group_norm(__global const float* X,
                         __global const float* gamma,
                         __global const float* beta,
                         __global float* Y,
                         const int N, const int C, const int H, const int W,
                         const int num_groups, const float eps) {
    int n = get_global_id(2);
    int c = get_global_id(1);
    int hw = get_global_id(0);

    if (n >= N || c >= C || hw >= H * W) return;

    int group_size = C / num_groups;
    int g = c / group_size;
    int c_start = g * group_size;
    int c_end = c_start + group_size;

    // compute mean and variance across channels in the group for this (n, hw)
    float sum = 0.0f;
    float sq_sum = 0.0f;
    int idx_base = n * C * H * W + hw; // position within spatial index
    for (int cc = c_start; cc < c_end; ++cc) {
        int idx = idx_base + cc * H * W;
        float v = X[idx];
        sum += v;
        sq_sum += v * v;
    }
    float m = sum / (float)group_size;
    float var = sq_sum / (float)group_size - m * m;
    float inv_std = rsqrt(fabs(var) + eps);

    int idx = idx_base + c * H * W;
    float xn = (X[idx] - m) * inv_std;
    Y[idx] = gamma[c] * xn + beta[c];
}

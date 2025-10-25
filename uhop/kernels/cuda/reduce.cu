// uhop/kernels/cuda/reduce.cu
// Reduction kernels using atomic operations (simple, portable)

extern "C" __global__
void reduce_sum_atomic(const float* X, float* Out, int N) {
    __shared__ float block_sum;
    if (threadIdx.x == 0) block_sum = 0.0f;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < N) ? X[i] : 0.0f;
    atomicAdd(&block_sum, v);
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(Out, block_sum);
}

extern "C" __global__
void reduce_max_atomic(const float* X, float* Out, int N) {
    __shared__ float block_max;
    if (threadIdx.x == 0) block_max = -1e38f;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < N) ? X[i] : -1e38f;
    atomicMax((int*)&block_max, __float_as_int(v));
    __syncthreads();
    if (threadIdx.x == 0) {
        // CAS loop to update global max
        int* out_i = (int*)Out;
        int old = atomicAdd(out_i, 0);
        int newv;
        do {
            float f_old = __int_as_float(old);
            float f_new = fmaxf(f_old, block_max);
            newv = __float_as_int(f_new);
            int prev = atomicCAS(out_i, old, newv);
            if (prev == old) break;
            old = prev;
        } while (true);
    }
}

extern "C" __global__
void reduce_min_atomic(const float* X, float* Out, int N) {
    __shared__ float block_min;
    if (threadIdx.x == 0) block_min = 1e38f;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < N) ? X[i] : 1e38f;
    // atomicMin for floats via CAS
    int* bptr = (int*)&block_min;
    int old = atomicAdd(bptr, 0);
    int newv;
    do {
        float f_old = __int_as_float(old);
        float f_new = fminf(f_old, v);
        newv = __float_as_int(f_new);
        int prev = atomicCAS(bptr, old, newv);
        if (prev == old) break;
        old = prev;
    } while (true);
    __syncthreads();
    if (threadIdx.x == 0) {
        int* out_i = (int*)Out;
        int oldg = atomicAdd(out_i, 0);
        int newg;
        do {
            float f_old = __int_as_float(oldg);
            float f_new = fminf(f_old, block_min);
            newg = __float_as_int(f_new);
            int prev = atomicCAS(out_i, oldg, newg);
            if (prev == oldg) break;
            oldg = prev;
        } while (true);
    }
}

extern "C" __global__
void reduce_norm_atomic(const float* X, float* Out, int N) {
    __shared__ float block_sum;
    if (threadIdx.x == 0) block_sum = 0.0f;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < N) ? X[i] : 0.0f;
    atomicAdd(&block_sum, v * v);
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(Out, block_sum);
}

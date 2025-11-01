__kernel void generated_relu(const int N, __global const float *X,
                             __global float *Y) {
  int i = get_global_id(0);
  if (i < N) {
    Y[i] = max(X[i], 0.0f);
  }
}
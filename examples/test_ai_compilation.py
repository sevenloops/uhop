"""
Test script for AI kernel compilation and profiling integration.
"""

from uhop.ai_codegen.compiler import KernelCompiler, KernelValidator


def test_compiler_initialization():
    """Test that the compiler can initialize properly."""
    print("=== Testing Compiler Initialization ===")

    compiler = KernelCompiler()
    print(f"Backend: {compiler.backend}")
    print(f"Context available: {compiler.context is not None}")
    print(f"Queue available: {compiler.queue is not None}")

    return compiler


def test_kernel_validation():
    """Test kernel validation with a simple matmul kernel."""
    print("\n=== Testing Kernel Validation ===")

    # Simple OpenCL matmul kernel
    kernel_code = """
    __kernel void simple_matmul(const int M, const int N, const int K,
                                __global const float* A,
                                __global const float* B,
                                __global float* C) {
        int i = get_global_id(0);
        int j = get_global_id(1);

        if (i < M && j < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    """

    validator = KernelValidator()
    input_shapes = [(128, 256), (256, 64)]

    print("Testing kernel compilation and validation...")
    result = validator.validate_and_profile_kernel(
        kernel_code=kernel_code, kernel_name="simple_matmul", operation="matmul", input_shapes=input_shapes
    )

    print(f"Compile success: {result['compile_success']}")
    print(f"Validation success: {result['validation_success']}")

    if result["compile_success"]:
        print(f"Execution time: {result['execution_time_ms']:.2f} ms")
        print(f"GFLOPS: {result['gflops']:.2f}")
        print(f"Bandwidth: {result['bandwidth_gbs']:.2f} GB/s")
        print(f"Max absolute error: {result['max_abs_error']:.2e}")
        print(f"Max relative error: {result['max_rel_error']:.2e}")

    if result.get("errors"):
        print(f"Errors: {result['errors']}")

    return result


def test_invalid_kernel():
    """Test handling of invalid kernel code."""
    print("\n=== Testing Invalid Kernel Handling ===")

    # Invalid kernel code (syntax error)
    kernel_code = """
    __kernel void invalid_kernel(const int M, const int N, const int K,
                                __global const float* A,
                                __global const float* B,
                                __global float* C) {
        int i = get_global_id(0);
        int j = get_global_id(1);

        if (i < M && j < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];  // Missing semicolon
            }
            C[i * N + j] = sum
        }
    }
    """

    validator = KernelValidator()
    input_shapes = [(32, 32), (32, 32)]

    result = validator.validate_and_profile_kernel(
        kernel_code=kernel_code, kernel_name="invalid_kernel", operation="matmul", input_shapes=input_shapes
    )

    print(f"Compile success: {result['compile_success']}")
    print(f"Validation success: {result['validation_success']}")

    if not result["compile_success"]:
        print("✓ Correctly detected compilation failure")
        if result.get("errors"):
            print(f"Error details: {result['errors']}")

    return result


def test_performance_comparison():
    """Compare performance of different kernel implementations."""
    print("\n=== Testing Performance Comparison ===")

    # Naive implementation
    naive_kernel = """
    __kernel void naive_matmul(const int M, const int N, const int K,
                              __global const float* A,
                              __global const float* B,
                              __global float* C) {
        int i = get_global_id(0);
        int j = get_global_id(1);

        if (i < M && j < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    """

    # Tiled implementation (simplified)
    tiled_kernel = """
    __kernel void tiled_matmul(const int M, const int N, const int K,
                              __global const float* A,
                              __global const float* B,
                              __global float* C) {
        int row = get_global_id(0);
        int col = get_global_id(1);

        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    """

    validator = KernelValidator()
    input_shapes = [(256, 256), (256, 256)]

    print("Testing naive kernel...")
    naive_result = validator.validate_and_profile_kernel(
        kernel_code=naive_kernel, kernel_name="naive_matmul", operation="matmul", input_shapes=input_shapes
    )

    print("Testing tiled kernel...")
    tiled_result = validator.validate_and_profile_kernel(
        kernel_code=tiled_kernel, kernel_name="tiled_matmul", operation="matmul", input_shapes=input_shapes
    )

    if naive_result["compile_success"] and tiled_result["compile_success"]:
        print("\nPerformance Comparison:")
        print(f"Naive kernel: {naive_result['execution_time_ms']:.2f} ms, {naive_result['gflops']:.2f} GFLOPS")
        print(f"Tiled kernel:  {tiled_result['execution_time_ms']:.2f} ms, {tiled_result['gflops']:.2f} GFLOPS")

        if naive_result["execution_time_ms"] > 0 and tiled_result["execution_time_ms"] > 0:
            speedup = naive_result["execution_time_ms"] / tiled_result["execution_time_ms"]
            print(f"Speedup: {speedup:.2f}x")


def main():
    """Run all compilation tests."""
    print("AI Kernel Compilation and Profiling Test")
    print("=" * 50)

    # Test 1: Compiler initialization
    compiler = test_compiler_initialization()

    # Test 2: Kernel validation
    validation_result = test_kernel_validation()

    # Test 3: Invalid kernel handling
    invalid_result = test_invalid_kernel()

    # Test 4: Performance comparison (if compilation worked)
    if validation_result["compile_success"]:
        test_performance_comparison()

    print("\n" + "=" * 50)
    print("Compilation testing completed!")

    # Summary
    print("\nSummary:")
    print(f"- Compiler initialized: {compiler.context is not None}")
    print(f"- Valid kernel compiled: {validation_result['compile_success']}")
    print(f"- Invalid kernel detected: {not invalid_result['compile_success']}")

    if validation_result["compile_success"]:
        print("\nNext steps:")
        print("1. Test with AI-generated kernels")
        print("2. Integrate with full AI pipeline")
        print("3. Add more performance metrics")


if __name__ == "__main__":
    main()

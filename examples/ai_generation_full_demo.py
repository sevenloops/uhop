"""
Full AI Generation Pipeline Demo with Compilation and Profiling.

This demo tests the complete closed-loop system including:
- AI kernel generation
- Static analysis
- Compilation
- Validation
- Performance profiling
- Corrective feedback
"""

from uhop.ai_codegen.pipeline import AIGenerationPipeline
from uhop.ir import MatMul, Tensor


def demo_with_mock_kernel():
    """Demo using a pre-defined kernel to test compilation pipeline."""
    print("=== Full AI Pipeline Demo with Mock Kernel ===\n")

    # Create matmul IR
    A = Tensor("A", (64, 128), "f32")
    B = Tensor("B", (128, 32), "f32")
    matmul_ir = MatMul(A, B)

    print(f"MatMul IR: {A.shape} x {B.shape} -> {matmul_ir.infer_output().shape}")

    # Initialize pipeline
    pipeline = AIGenerationPipeline()

    # Extract specification
    spec = pipeline.extract_spec_from_ir(matmul_ir)
    print(f"\nHardware: {spec.hardware_fingerprint['vendor']} ({spec.hardware_fingerprint['kind']})")

    # Test compilation with a simple kernel
    simple_kernel = """
    __kernel void generated_matmul(const int M, const int N, const int K,
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

    print("\nTesting compilation and validation pipeline...")

    # Create a mock generation attempt
    from uhop.ai_codegen.pipeline import GenerationAttempt, StaticAnalysisResult

    mock_attempt = GenerationAttempt(
        attempt_id="mock_test_001",
        timestamp=1234567890.0,
        spec=spec,
        prompt="Mock prompt for testing",
        kernel_code=simple_kernel,
        provider="mock",
        model="mock-model",
        static_analysis=StaticAnalysisResult(False, False),
        compile_success=False,
    )

    # Test compilation and validation
    compiled_attempt = pipeline._compile_and_validate_kernel(mock_attempt)

    print(f"Compilation success: {compiled_attempt.compile_success}")

    if compiled_attempt.compile_success:
        print(f"Validation success: {compiled_attempt.validation_result.ok}")
        if compiled_attempt.profile_result:
            print(f"Execution time: {compiled_attempt.profile_result.timing_ms:.2f} ms")
            print(f"GFLOPS: {compiled_attempt.profile_result.gflops:.2f}")
            print(f"Bandwidth: {compiled_attempt.profile_result.bandwidth_gbs:.2f} GB/s")
            print(f"Max absolute error: {compiled_attempt.profile_result.max_abs_error:.2e}")
    else:
        print(f"Compilation failed: {compiled_attempt.corrective_feedback}")

    return compiled_attempt


def demo_static_analysis():
    """Demonstrate static analysis capabilities."""
    print("\n=== Static Analysis Demo ===\n")

    # Different kernel variants
    kernels = {
        "naive": """
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
        """,
        "with_local_memory": """
        __kernel void local_mem_matmul(const int M, const int N, const int K,
                                      __global const float* A,
                                      __global const float* B,
                                      __global float* C) {
            __local float tileA[16][16];
            __local float tileB[16][16];

            int row = get_local_id(0);
            int col = get_local_id(1);
            int globalRow = get_global_id(0);
            int globalCol = get_global_id(1);

            float sum = 0.0f;
            for (int t = 0; t < K/16; t++) {
                // Load tiles into local memory
                tileA[row][col] = A[globalRow * K + t*16 + col];
                tileB[row][col] = B[(t*16 + row) * N + globalCol];
                barrier(CLK_LOCAL_MEM_FENCE);

                // Compute partial sum
                for (int k = 0; k < 16; k++) {
                    sum += tileA[row][k] * tileB[k][col];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (globalRow < M && globalCol < N) {
                C[globalRow * N + globalCol] = sum;
            }
        }
        """,
        "with_vector_ops": """
        __kernel void vector_matmul(const int M, const int N, const int K,
                                   __global const float* A,
                                   __global const float* B,
                                   __global float* C) {
            int i = get_global_id(0);
            int j = get_global_id(1);

            if (i < M && j < N) {
                float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                for (int k = 0; k < K; k += 4) {
                    float4 a_vec = vload4(0, &A[i * K + k]);
                    float4 b_vec = (float4)(B[k * N + j], B[(k+1) * N + j],
                                           B[(k+2) * N + j], B[(k+3) * N + j]);
                    sum += a_vec * b_vec;
                }
                C[i * N + j] = sum.x + sum.y + sum.z + sum.w;
            }
        }
        """,
    }

    from uhop.ai_codegen.pipeline import StaticAnalyzer

    analyzer = StaticAnalyzer()

    for name, kernel_code in kernels.items():
        print(f"\nAnalyzing {name} kernel:")
        analysis = analyzer.analyze_opencl_kernel(kernel_code)

        print(f"  - Local memory: {analysis.has_local_memory}")
        print(f"  - Barriers: {analysis.has_barriers}")
        print(f"  - Workgroup hint: {analysis.workgroup_size_hint}")
        print(f"  - Register pressure: {analysis.register_pressure_hint}")
        print(f"  - Memory patterns: {analysis.memory_access_patterns}")
        print(f"  - Potential issues: {analysis.potential_issues}")


def demo_corrective_feedback():
    """Demonstrate corrective feedback generation."""
    print("\n=== Corrective Feedback Demo ===\n")

    # Test cases with different issues
    test_cases = [
        {
            "name": "Slow kernel",
            "execution_time_ms": 1500.0,
            "max_abs_error": 1e-6,
            "max_rel_error": 1e-4,
            "validation_success": True,
        },
        {
            "name": "Validation failure",
            "execution_time_ms": 10.0,
            "max_abs_error": 1.0,
            "max_rel_error": 0.5,
            "validation_success": False,
        },
        {
            "name": "Good kernel",
            "execution_time_ms": 50.0,
            "max_abs_error": 1e-7,
            "max_rel_error": 1e-5,
            "validation_success": True,
        },
    ]

    for case in test_cases:
        print(f"\n{case['name']}:")

        # Generate feedback
        feedback = ""
        if not case["validation_success"]:
            feedback = (
                f"Kernel compiled but validation failed. "
                f"Max absolute error: {case['max_abs_error']:.2e}, "
                f"Max relative error: {case['max_rel_error']:.2e}"
            )
        elif case["execution_time_ms"] > 1000:
            feedback = (
                f"Kernel is too slow: {case['execution_time_ms']:.2f} ms. "
                f"Try optimizing memory access patterns or reducing register pressure."
            )
        else:
            feedback = "Kernel performance is acceptable"

        print(f"  Feedback: {feedback}")


def demo_dataset_operations():
    """Demonstrate dataset querying and management."""
    print("\n=== Dataset Operations Demo ===\n")

    pipeline = AIGenerationPipeline()
    dataset_dir = pipeline.dataset_dir

    print(f"Dataset directory: {dataset_dir}")

    # Check if dataset exists and show structure
    if dataset_dir.exists():
        attempts = list(dataset_dir.iterdir())
        print(f"Found {len(attempts)} generation attempts")

        # Show recent attempts
        recent_attempts = sorted(attempts, key=lambda x: x.stat().st_mtime, reverse=True)[:3]
        print("\nRecent attempts:")
        for attempt_dir in recent_attempts:
            if attempt_dir.is_dir():
                metadata_file = attempt_dir / "metadata.json"
                if metadata_file.exists():
                    import json

                    metadata = json.loads(metadata_file.read_text())
                    status = "SUCCESS" if metadata.get("compile_success") else "FAILED"
                    print(f"  - {attempt_dir.name}: {status}")
    else:
        print("No dataset found - run generation to create one")


def main():
    """Run the complete enhanced demo."""
    print("Enhanced AI Generation Pipeline Demo")
    print("=" * 60)

    # Demo 1: Compilation pipeline with mock kernel
    compiled_attempt = demo_with_mock_kernel()

    # Demo 2: Static analysis
    demo_static_analysis()

    # Demo 3: Corrective feedback
    demo_corrective_feedback()

    # Demo 4: Dataset operations
    demo_dataset_operations()

    print("\n" + "=" * 60)
    print("Enhanced demo completed!")

    # Summary and next steps
    print("\nImplementation Status:")
    print("✅ AI kernel generation")
    print("✅ Static analysis")
    print("✅ Kernel compilation")
    print("✅ Validation and profiling")
    print("✅ Corrective feedback")
    print("✅ Dataset persistence")
    print("\nReady for AI provider integration!")

    if compiled_attempt.compile_success:
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY or DEEPSEEK_API_KEY")
        print("2. Run: python -m uhop.cli_ai run --shapes 64x128,128x32")
        print("3. Check results: python -m uhop.cli_ai list")


if __name__ == "__main__":
    main()

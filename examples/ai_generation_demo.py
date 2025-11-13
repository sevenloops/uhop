"""
Demo script for AI kernel generation pipeline.

This script demonstrates the closed-loop AI generation system for matmul kernels.
"""

import os
import sys
from pathlib import Path
from uhop.ai_codegen.pipeline import AIGenerationPipeline
from uhop.ir import MatMul, Tensor

# Add uhop to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_basic_generation():
    """Demonstrate basic AI kernel generation."""
    print("=== AI Kernel Generation Demo ===\n")

    # Create a simple matmul IR
    A = Tensor("A", (128, 256), "f32")
    B = Tensor("B", (256, 64), "f32")
    matmul_ir = MatMul(A, B)

    print(f"MatMul IR: {A.shape} x {B.shape} -> {matmul_ir.infer_output().shape}")

    # Initialize pipeline
    pipeline = AIGenerationPipeline()

    # Extract specification
    spec = pipeline.extract_spec_from_ir(matmul_ir)
    print("\nExtracted Spec:")
    print(f"  Operation: {spec.operation}")
    print(f"  Input shapes: {spec.input_shapes}")
    print(f"  Output shape: {spec.output_shape}")
    print(f"  Hardware: {spec.hardware_fingerprint['vendor']} ({spec.hardware_fingerprint['kind']})")
    print(f"  IR Hash: {spec.ir_hash[:16]}...")

    # Compose prompt
    prompt = pipeline.compose_prompt(spec)
    print(f"\nGenerated Prompt ({len(prompt)} chars):")
    print("-" * 50)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

    return pipeline, matmul_ir


def demo_single_generation(pipeline, matmul_ir):
    """Demonstrate single kernel generation attempt."""
    print("\n=== Single Generation Attempt ===\n")

    # Generate a single kernel
    spec = pipeline.extract_spec_from_ir(matmul_ir)

    # Check if we have API keys
    has_openai = "OPENAI_API_KEY" in os.environ
    has_deepseek = "DEEPSEEK_API_KEY" in os.environ

    print("Available providers:")
    print(f"  OpenAI: {'YES' if has_openai else 'NO (set OPENAI_API_KEY)'}")
    print(f"  DeepSeek: {'YES' if has_deepseek else 'NO (set DEEPSEEK_API_KEY)'}")

    if not has_openai and not has_deepseek:
        print("\nNo API keys found. Please set OPENAI_API_KEY or DEEPSEEK_API_KEY.")
        print("Demo will show pipeline structure without actual generation.")
        return

    # Try generation with available provider
    provider = "openai" if has_openai else "deepseek"

    try:
        attempt = pipeline.generate_kernel(spec, provider=provider, max_retries=2)

        print("\nGeneration Result:")
        print(f"  Attempt ID: {attempt.attempt_id}")
        print(f"  Provider: {attempt.provider}")
        print(f"  Compile Success: {attempt.compile_success}")
        print(f"  Retry Count: {attempt.retry_count}")

        if attempt.compile_success:
            print(f"\nGenerated Kernel ({len(attempt.kernel_code)} chars):")
            print("-" * 50)
            print(attempt.kernel_code[:500] + "..." if len(attempt.kernel_code) > 500 else attempt.kernel_code)

            print("\nStatic Analysis:")
            print(f"  Has local memory: {attempt.static_analysis.has_local_memory}")
            print(f"  Has barriers: {attempt.static_analysis.has_barriers}")
            print(f"  Workgroup hint: {attempt.static_analysis.workgroup_size_hint}")
            print(f"  Register pressure: {attempt.static_analysis.register_pressure_hint}")
            print(f"  Memory patterns: {attempt.static_analysis.memory_access_patterns}")
            print(f"  Potential issues: {attempt.static_analysis.potential_issues}")
        else:
            print(f"\nGeneration failed: {attempt.corrective_feedback}")

    except Exception as e:
        print(f"\nGeneration failed with error: {e}")


def demo_pipeline_run(pipeline, matmul_ir):
    """Demonstrate full pipeline run."""
    print("\n=== Full Pipeline Run ===\n")

    # Check if we have API keys
    has_openai = "OPENAI_API_KEY" in os.environ
    has_deepseek = "DEEPSEEK_API_KEY" in os.environ

    if not has_openai and not has_deepseek:
        print("No API keys available. Skipping pipeline run.")
        return

    try:
        # Run full pipeline with limited attempts
        attempts = pipeline.run_pipeline(matmul_ir, max_attempts=2)

        print(f"Pipeline completed with {len(attempts)} attempts:")

        successful = [a for a in attempts if a.compile_success]
        failed = [a for a in attempts if not a.compile_success]

        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")

        if successful:
            print("\nSuccessful attempts:")
            for attempt in successful:
                print(f"  - {attempt.attempt_id} (retries: {attempt.retry_count})")

        if failed:
            print("\nFailed attempts:")
            for attempt in failed:
                print(f"  - {attempt.attempt_id}: {attempt.corrective_feedback}")

    except Exception as e:
        print(f"Pipeline run failed: {e}")


def demo_dataset_query(pipeline):
    """Demonstrate dataset querying."""
    print("\n=== Dataset Query Demo ===\n")

    dataset_dir = pipeline.dataset_dir

    if not dataset_dir.exists():
        print(f"No dataset found at {dataset_dir}")
        return

    # Count attempts
    attempts = []
    for attempt_dir in dataset_dir.iterdir():
        if attempt_dir.is_dir():
            metadata_file = attempt_dir / "metadata.json"
            if metadata_file.exists():
                attempts.append(attempt_dir.name)

    print(f"Dataset contains {len(attempts)} generation attempts")

    if attempts:
        print("\nRecent attempts:")
        for attempt_id in sorted(attempts)[-5:]:  # Show last 5
            print(f"  - {attempt_id}")


def main():
    """Run the complete demo."""
    print("UHOP AI Kernel Generation Pipeline Demo")
    print("=" * 50)

    # Demo 1: Basic generation setup
    pipeline, matmul_ir = demo_basic_generation()

    # Demo 2: Single generation attempt
    demo_single_generation(pipeline, matmul_ir)

    # Demo 3: Full pipeline run
    demo_pipeline_run(pipeline, matmul_ir)

    # Demo 4: Dataset query
    demo_dataset_query(pipeline)

    print("\n" + "=" * 50)
    print("Demo completed!")
    print(f"Dataset location: {pipeline.dataset_dir}")
    print("\nNext steps:")
    print("1. Set OPENAI_API_KEY or DEEPSEEK_API_KEY environment variables")
    print("2. Run 'python -m uhop.cli_ai run --shapes 256x512,512x128'")
    print("3. Run 'python -m uhop.cli_ai list' to see generated attempts")


if __name__ == "__main__":
    main()

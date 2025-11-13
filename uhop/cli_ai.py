"""
CLI for AI kernel generation pipeline.

Provides commands for running the closed-loop AI generation system.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional  # noqa: F401

from uhop.ai_codegen.pipeline import AIGenerationPipeline
from uhop.ir import MatMul, Tensor


def create_matmul_ir_from_shapes(shape_str: str) -> MatMul:
    """Create MatMul IR from shape string in format MxK,KxN."""
    shape_parts = shape_str.split(",")
    if len(shape_parts) != 2:
        raise ValueError("Shapes must be in format MxK,KxN")

    m, k = map(int, shape_parts[0].split("x"))
    k2, n = map(int, shape_parts[1].split("x"))

    if k != k2:
        raise ValueError("Inner dimensions must match")

    A = Tensor("A", (m, k), "f32")
    B = Tensor("B", (k, n), "f32")
    return MatMul(A, B)


def run_generation(args):
    """Run AI generation loop for matmul."""
    try:
        matmul_ir = create_matmul_ir_from_shapes(args.shapes)
    except ValueError as e:
        print(f"Error parsing shapes: {e}")
        return 1

    pipeline = AIGenerationPipeline(dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None)

    print(f"Starting AI generation for matmul {args.shapes}")
    print(f"Provider: {args.provider}")
    print(f"Max attempts: {args.max_attempts}")
    print(f"Dataset directory: {pipeline.dataset_dir}")
    print("-" * 50)

    attempts = pipeline.run_pipeline(matmul_ir, max_attempts=args.max_attempts)

    print(f"\nGenerated {len(attempts)} attempts")

    successful = [a for a in attempts if a.compile_success]
    failed = [a for a in attempts if not a.compile_success]

    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\nSuccessful attempts:")
        for attempt in successful:
            print(f"  - {attempt.attempt_id} (Retries: {attempt.retry_count})")

    if failed:
        print("\nFailed attempts:")
        for attempt in failed:
            print(f"  - {attempt.attempt_id}: {attempt.corrective_feedback}")

    return 0


def list_attempts(args):
    """List generation attempts from dataset."""
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path.home() / ".uhop_ai_dataset"

    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return 1

    attempts = []
    for attempt_dir in dataset_dir.iterdir():
        if attempt_dir.is_dir():
            metadata_file = attempt_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text())
                    attempts.append(metadata)
                except Exception as e:
                    print(f"Error reading {metadata_file}: {e}")

    if not attempts:
        print("No generation attempts found")
        return 0

    print(f"Found {len(attempts)} generation attempts:")
    print("-" * 80)

    for attempt in sorted(attempts, key=lambda x: x.get("timestamp", 0)):
        attempt_id = attempt.get("attempt_id", "unknown")
        operation = attempt.get("spec", {}).get("operation", "unknown")
        provider = attempt.get("provider", "unknown")
        compile_success = attempt.get("compile_success", False)
        retry_count = attempt.get("retry_count", 0)

        status = "SUCCESS" if compile_success else "FAILED"
        print(f"{attempt_id:30} {operation:10} {provider:10} {status:8} (retries: {retry_count})")

    return 0


def show_attempt(args):
    """Show details of a specific generation attempt."""
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path.home() / ".uhop_ai_dataset"
    attempt_dir = dataset_dir / args.attempt_id

    if not attempt_dir.exists():
        print(f"Attempt not found: {args.attempt_id}")
        return 1

    metadata_file = attempt_dir / "metadata.json"
    kernel_file = attempt_dir / "kernel.cl"
    prompt_file = attempt_dir / "prompt.txt"

    if not metadata_file.exists():
        print(f"Metadata not found for attempt: {args.attempt_id}")
        return 1

    try:
        metadata = json.loads(metadata_file.read_text())
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return 1

    print(f"Attempt ID: {metadata.get('attempt_id')}")
    print(f"Operation: {metadata.get('spec', {}).get('operation')}")
    print(f"Provider: {metadata.get('provider')}")
    print(f"Model: {metadata.get('model')}")
    print(f"Timestamp: {metadata.get('timestamp')}")
    print(f"Compile Success: {metadata.get('compile_success')}")
    print(f"Retry Count: {metadata.get('retry_count')}")

    if kernel_file.exists():
        kernel_code = kernel_file.read_text()
        print(f"\nKernel Code ({len(kernel_code)} chars):")
        print("-" * 50)
        print(kernel_code[:500] + "..." if len(kernel_code) > 500 else kernel_code)

    if prompt_file.exists():
        prompt = prompt_file.read_text()
        print(f"\nPrompt ({len(prompt)} chars):")
        print("-" * 50)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

    static_analysis = metadata.get("static_analysis", {})
    if static_analysis:
        print("\nStatic Analysis:")
        print(f"  - Has local memory: {static_analysis.get('has_local_memory')}")
        print(f"  - Has barriers: {static_analysis.get('has_barriers')}")
        print(f"  - Workgroup hint: {static_analysis.get('workgroup_size_hint')}")
        print(f"  - Register pressure: {static_analysis.get('register_pressure_hint')}")
        print(f"  - Memory patterns: {static_analysis.get('memory_access_patterns')}")
        print(f"  - Potential issues: {static_analysis.get('potential_issues')}")

    return 0


def query_dataset(args):
    """Query dataset by device or specification."""
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path.home() / ".uhop_ai_dataset"

    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return 1

    matching_attempts = []

    for attempt_dir in dataset_dir.iterdir():
        if attempt_dir.is_dir():
            metadata_file = attempt_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text())

                    # Apply filters
                    matches = True

                    if args.device:
                        hw_fingerprint = metadata.get("spec", {}).get("hardware_fingerprint", {})
                        device_vendor = hw_fingerprint.get("vendor", "").lower()
                        if args.device.lower() not in device_vendor:
                            matches = False

                    if args.operation:
                        operation = metadata.get("spec", {}).get("operation", "").lower()
                        if args.operation.lower() != operation:
                            matches = False

                    if args.success_only and not metadata.get("compile_success", False):
                        matches = False

                    if matches:
                        matching_attempts.append(metadata)

                except Exception as e:
                    print(f"Error reading {metadata_file}: {e}")

    if not matching_attempts:
        print("No matching attempts found")
        return 0

    print(f"Found {len(matching_attempts)} matching attempts:")
    print("-" * 80)

    for attempt in sorted(matching_attempts, key=lambda x: x.get("timestamp", 0)):
        attempt_id = attempt.get("attempt_id", "unknown")
        operation = attempt.get("spec", {}).get("operation", "unknown")
        provider = attempt.get("provider", "unknown")
        device = attempt.get("spec", {}).get("hardware_fingerprint", {}).get("vendor", "unknown")
        compile_success = attempt.get("compile_success", False)

        status = "SUCCESS" if compile_success else "FAILED"
        print(f"{attempt_id:30} {operation:10} {provider:10} {device:10} {status:8}")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="UHOP AI Kernel Generation Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run generation command
    run_parser = subparsers.add_parser("run", help="Run AI generation loop")
    run_parser.add_argument(
        "--shapes", type=str, required=True, help="Matrix shapes in format MxK,KxN (e.g., 1024x512,512x2048)"
    )
    run_parser.add_argument("--max-attempts", type=int, default=5, help="Maximum number of generation attempts")
    run_parser.add_argument(
        "--provider", type=str, default="openai", choices=["openai", "deepseek"], help="AI provider to use"
    )
    run_parser.add_argument("--dataset-dir", type=str, help="Dataset directory for storing results")
    run_parser.set_defaults(func=run_generation)

    # List attempts command
    list_parser = subparsers.add_parser("list", help="List generation attempts")
    list_parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
    list_parser.set_defaults(func=list_attempts)

    # Show attempt command
    show_parser = subparsers.add_parser("show", help="Show details of a generation attempt")
    show_parser.add_argument("attempt_id", type=str, help="Attempt ID to show")
    show_parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
    show_parser.set_defaults(func=show_attempt)

    # Query dataset command
    query_parser = subparsers.add_parser("query", help="Query dataset by device or specification")
    query_parser.add_argument("--device", type=str, help="Filter by device vendor (e.g., nvidia, amd, intel)")
    query_parser.add_argument("--operation", type=str, help="Filter by operation (e.g., matmul, relu)")
    query_parser.add_argument("--success-only", action="store_true", help="Only show successful attempts")
    query_parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
    query_parser.set_defaults(func=query_dataset)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

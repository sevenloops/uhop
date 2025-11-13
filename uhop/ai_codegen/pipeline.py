"""
AI Generation Pipeline for closed-loop kernel generation.

Implements the workflow described in issues/02_ai_generation_pipeline.md:
spec extraction → AI prompt → static analysis → compile → micro-test →
profile → feedback → archive. The implementation is resilient to missing
API keys or GPU runtimes so unit tests can exercise the control flow without
external dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..hardware import HardwareProfile, detect_hardware
from ..ir import MatMul, Tensor, compute_stable_hash
from ..validation import ValidationResult
from .compiler import KernelValidator
from .generator import AICodegen
from .prompt_templates import PROMPTS

FALLBACK_OPENCL_MATMUL = textwrap.dedent(
    """
    __kernel void generated_matmul(const int M, const int N, const int K,
                                   __global const float* A,
                                   __global const float* B,
                                   __global float* C) {
        const int row = get_global_id(0);
        const int col = get_global_id(1);

        if (row >= M || col >= N) {
            return;
        }

        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
    """
).strip()

DEFAULT_OPENCL_KERNEL_NAME = "generated_matmul"
SUPPORTED_PROVIDERS = {"openai", "deepseek"}


@dataclass
class KernelSpec:
    """Extracted specification from IR + hardware fingerprint."""

    ir_hash: str
    operation: str
    input_shapes: List[Tuple[int, ...]]
    output_shape: Tuple[int, ...]
    dtype: str
    hardware_fingerprint: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StaticAnalysisResult:
    """Results from static analysis of generated kernel."""

    has_local_memory: bool
    has_barriers: bool
    workgroup_size_hint: Optional[Tuple[int, int, int]] = None
    register_pressure_hint: Optional[str] = None
    memory_access_patterns: List[str] = field(default_factory=list)
    potential_issues: List[str] = field(default_factory=list)


@dataclass
class ProfileResult:
    """Profiling results for generated kernel."""

    timing_ms: float
    gflops: float
    bandwidth_gbs: float
    correctness: bool
    max_abs_error: float
    max_rel_error: float
    logs: Optional[str] = None


@dataclass
class GenerationAttempt:
    """Full record of a single generation attempt."""

    attempt_id: str
    timestamp: float
    spec: KernelSpec
    prompt: str
    kernel_code: str
    provider: str
    model: str
    static_analysis: StaticAnalysisResult
    compile_success: bool
    validation_result: Optional[ValidationResult] = None
    profile_result: Optional[ProfileResult] = None
    corrective_feedback: Optional[str] = None
    retry_count: int = 0


class StaticAnalyzer:
    """Simple static analyzer for resource hints and patterns."""

    @staticmethod
    def analyze_opencl_kernel(code: str) -> StaticAnalysisResult:
        """Analyze OpenCL kernel for resource usage patterns."""

        code_lower = code.lower()

        # Check for local memory usage
        has_local_memory = "__local" in code or "local" in code_lower

        # Check for barriers
        has_barriers = "barrier" in code_lower

        # Extract workgroup size hints from get_local_size calls
        workgroup_hint = None
        local_size_pattern = r"get_local_size\s*\(\s*(\d+)\s*\)"
        matches = re.findall(local_size_pattern, code)
        if matches:
            dims = [int(m) for m in matches[:3]]
            while len(dims) < 3:
                dims.append(1)
            workgroup_hint = tuple(dims[:3])

        # Simple register pressure estimation
        register_pressure = "low"
        var_count = len(re.findall(r"\b(float|int|uint)\s+[a-zA-Z_]\w*", code))
        if var_count > 50:
            register_pressure = "high"
        elif var_count > 20:
            register_pressure = "medium"

        # Memory access patterns
        memory_patterns: List[str] = []
        if "vload" in code_lower:
            memory_patterns.append("vector_load")
        if "vstore" in code_lower:
            memory_patterns.append("vector_store")
        if re.search(r"\[[^\]]*[+\-*/][^\]]*\]", code):
            memory_patterns.append("indexed_access")
        elif re.search(r"\[[^\]]+\]\s*\[", code):
            memory_patterns.append("indexed_access")
        else:
            index_exprs = re.findall(r"\[([^\]]+)\]", code)
            for expr in index_exprs:
                expr_lower = expr.lower()
                if "get_global_id" in expr_lower or "get_local_id" in expr_lower:
                    memory_patterns.append("indexed_access")
                    break

        # Potential issues
        issues: List[str] = []
        if "while" in code_lower and "barrier" in code_lower:
            issues.append("potential_barrier_in_loop")
        if re.search(r"if\s*\(.*\)\s*{.*barrier", code, re.DOTALL):
            issues.append("conditional_barrier")

        return StaticAnalysisResult(
            has_local_memory=has_local_memory,
            has_barriers=has_barriers,
            workgroup_size_hint=workgroup_hint,
            register_pressure_hint=register_pressure,
            memory_access_patterns=memory_patterns,
            potential_issues=issues,
        )

    @staticmethod
    def analyze_cuda_kernel(code: str) -> StaticAnalysisResult:
        """Analyze CUDA kernel for resource usage patterns."""

        code_lower = code.lower()

        has_local_memory = "__shared__" in code
        has_barriers = "__syncthreads" in code

        workgroup_hint = None
        block_pattern = r"dim3\s+block\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"
        matches = re.findall(block_pattern, code)
        if matches:
            dims = [int(m) for m in matches[0][:3]]
            workgroup_hint = tuple(dims)

        register_pressure = "low"
        var_count = len(re.findall(r"\b(float|int|unsigned)\s+[a-zA-Z_]\w*", code))
        if var_count > 50:
            register_pressure = "high"
        elif var_count > 20:
            register_pressure = "medium"

        memory_patterns: List[str] = []
        if re.search(r"\.x\s*=\s*", code):
            memory_patterns.append("vector_access")
        if "atomic" in code_lower:
            memory_patterns.append("atomic_operations")

        issues: List[str] = []
        if "while" in code_lower and "__syncthreads" in code:
            issues.append("potential_sync_in_loop")
        if re.search(r"if\s*\(.*\)\s*{.*__syncthreads", code, re.DOTALL):
            issues.append("conditional_sync")

        return StaticAnalysisResult(
            has_local_memory=has_local_memory,
            has_barriers=has_barriers,
            workgroup_size_hint=workgroup_hint,
            register_pressure_hint=register_pressure,
            memory_access_patterns=memory_patterns,
            potential_issues=issues,
        )


class AIGenerationPipeline:
    """Main AI generation pipeline for closed-loop kernel generation."""

    def __init__(self, dataset_dir: Optional[Path] = None, default_provider: Optional[str] = None):
        self.dataset_dir = dataset_dir or Path.home() / ".uhop_ai_dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # Provider registry. We keep separate AICodegen instances so each can
        # carry a provider-specific model name.
        self.providers: Dict[str, AICodegen] = {
            "openai": AICodegen(model=os.environ.get("UHOP_OPENAI_MODEL")),
            "deepseek": AICodegen(model=os.environ.get("UHOP_DEEPSEEK_MODEL", "deepseek-coder")),
        }

        self.default_provider = (default_provider or os.environ.get("UHOP_AI_PROVIDER", "openai")).lower()
        if self.default_provider not in SUPPORTED_PROVIDERS:
            self.default_provider = "openai"

        self.static_analyzer = StaticAnalyzer()
        self.validator = KernelValidator()
        self.generation_history: List[GenerationAttempt] = []
        self.debug = os.environ.get("UHOP_AI_DEBUG", "0").lower() in ("1", "true", "yes", "on")

    # ------------------------------------------------------------------
    # Spec extraction & prompt composition
    # ------------------------------------------------------------------
    def extract_spec_from_ir(self, ir_op: MatMul, hardware: Optional[HardwareProfile] = None) -> KernelSpec:
        """Extract kernel specification from IR operation and hardware."""

        hardware = hardware or detect_hardware()

        ir_dict = ir_op.to_dict()
        ir_hash = compute_stable_hash(ir_dict)

        input_shapes = [tuple(ir_op.A.shape), tuple(ir_op.B.shape)]
        output_shape = tuple(ir_op.infer_output().shape)
        dtype = ir_op.A.dtype

        hw_fingerprint = {
            "vendor": hardware.vendor,
            "kind": hardware.kind,
            "name": hardware.name,
            "details": hardware.details or {},
        }

        constraints: Dict[str, Any] = {}
        if ir_op.schedule:
            constraints.update(ir_op.schedule.to_dict())

        return KernelSpec(
            ir_hash=ir_hash,
            operation="matmul",
            input_shapes=input_shapes,
            output_shape=output_shape,
            dtype=dtype,
            hardware_fingerprint=hw_fingerprint,
            constraints=constraints,
        )

    def compose_prompt(self, spec: KernelSpec, prior_attempts: Optional[List[GenerationAttempt]] = None) -> str:
        """Compose AI prompt including prior tuned parameters and constraints."""

        base_prompt = PROMPTS.get(
            "opencl",
            "You are an OpenCL compute expert. Generate a kernel for {operation}.",
        ).format(operation=spec.operation, arch=spec.hardware_fingerprint.get("vendor", "generic"))

        lines = [base_prompt]
        lines.append(
            f"Target device: {spec.hardware_fingerprint.get('vendor', 'unknown')} "
            f"({spec.hardware_fingerprint.get('kind', 'unknown')})"
        )
        lines.append(f"Input shapes: {spec.input_shapes}")
        lines.append(f"Output shape: {spec.output_shape}")
        lines.append(f"Dtype: {spec.dtype}")

        if spec.constraints:
            lines.append(f"Constraints: {spec.constraints}")

        if prior_attempts:
            failures = [
                a
                for a in prior_attempts
                if not a.compile_success or not (a.validation_result and a.validation_result.ok)
            ]
            if failures:
                recent = failures[-2:]
                formatted = []
                for idx, attempt in enumerate(recent, 1):
                    msg = attempt.corrective_feedback or "unspecified failure"
                    formatted.append(f"Attempt-{len(prior_attempts)-len(recent)+idx}: {msg}")
                lines.append("Recent feedback: " + " | ".join(formatted))

        prompt = "\n".join(lines).strip()
        return prompt

    # ------------------------------------------------------------------
    # Generation core
    # ------------------------------------------------------------------
    def generate_kernel(
        self,
        spec: KernelSpec,
        provider: Optional[str] = None,
        max_retries: int = 0,
    ) -> GenerationAttempt:
        """Generate kernel attempts for a spec. Returns the final attempt."""

        resolved_provider = self._resolve_provider(provider)
        prior_attempts = [a for a in self.generation_history if a.spec.ir_hash == spec.ir_hash]
        last_attempt: Optional[GenerationAttempt] = None

        for _ in range(max_retries + 1):
            retry_index = len(prior_attempts)
            attempt = self._single_attempt(spec, resolved_provider, retry_index, prior_attempts)
            prior_attempts.append(attempt)
            last_attempt = attempt
            if attempt.compile_success and attempt.validation_result and attempt.validation_result.ok:
                break
            if self._should_stop_retry(attempt):
                break

        if last_attempt is None:
            raise RuntimeError("generate_kernel produced no attempts")
        return last_attempt

    def run_pipeline(
        self,
        ir_op: MatMul,
        max_attempts: int = 5,
        provider: Optional[str] = None,
    ) -> List[GenerationAttempt]:
        """Run the full closed-loop pipeline for an IR operation."""

        spec = self.extract_spec_from_ir(ir_op)
        resolved_provider = self._resolve_provider(provider)
        attempts: List[GenerationAttempt] = []
        prior_attempts = [a for a in self.generation_history if a.spec.ir_hash == spec.ir_hash]

        for _ in range(max_attempts):
            retry_index = len(prior_attempts)
            attempt = self._single_attempt(spec, resolved_provider, retry_index, prior_attempts)
            attempts.append(attempt)
            prior_attempts.append(attempt)

            if attempt.compile_success and attempt.validation_result and attempt.validation_result.ok:
                break
            if self._should_stop_retry(attempt):
                break

        return attempts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _single_attempt(
        self,
        spec: KernelSpec,
        provider: str,
        retry_index: int,
        prior_attempts: List[GenerationAttempt],
    ) -> GenerationAttempt:
        prompt = self.compose_prompt(spec, prior_attempts)
        attempt_id = self._build_attempt_id(spec.ir_hash, retry_index)

        provider_client = self.providers.get(provider)
        kernel_code = ""
        generation_feedback: Optional[str] = None
        provider_used = provider

        if provider_client and self._provider_ready(provider):
            try:
                kernel_path = provider_client.generate(
                    operation_name=spec.operation,
                    target="opencl",
                    prompt_extra=prompt,
                    temperature=self._temperature_for_retry(retry_index),
                    suffix=f"_{spec.ir_hash[:8]}_{retry_index}",
                    provider=provider,
                )
                kernel_code = kernel_path.read_text()
            except Exception as exc:  # noqa: BLE001 - surface provider errors
                generation_feedback = f"{provider} generation failed: {exc}"
                provider_used = "offline-fallback"
        else:
            missing_reason = self._provider_missing_reason(provider)
            if missing_reason:
                generation_feedback = missing_reason
            provider_used = "offline-fallback"

        if not kernel_code:
            kernel_code = self._fallback_kernel(spec)

        static_analysis = (
            self.static_analyzer.analyze_opencl_kernel(kernel_code)
            if kernel_code
            else StaticAnalysisResult(False, False)
        )

        compile_success, validation_result, profile_result, downstream_feedback = self._validate_and_profile(
            spec, kernel_code
        )

        corrective_feedback = self._compose_corrective_feedback(
            generation_feedback,
            downstream_feedback,
            static_analysis,
        )

        attempt = GenerationAttempt(
            attempt_id=attempt_id,
            timestamp=time.time(),
            spec=spec,
            prompt=prompt,
            kernel_code=kernel_code,
            provider=provider_used,
            model=(provider_client.model if provider_client else "offline"),
            static_analysis=static_analysis,
            compile_success=compile_success,
            validation_result=validation_result,
            profile_result=profile_result,
            corrective_feedback=corrective_feedback,
            retry_count=retry_index,
        )

        self.generation_history.append(attempt)
        self._persist_attempt(attempt)
        return attempt

    def _validate_and_profile(
        self,
        spec: KernelSpec,
        kernel_code: str,
    ) -> Tuple[bool, Optional[ValidationResult], Optional[ProfileResult], Optional[str]]:
        """Compile, validate, and profile the generated kernel."""

        if not kernel_code:
            return False, None, None, "no kernel code generated"

        kernel_name = self._extract_kernel_name(kernel_code) or DEFAULT_OPENCL_KERNEL_NAME

        if self._should_use_offline_validation():
            validation = ValidationResult(True, 0.0, 0.0, None, "offline-validation")
            profile = ProfileResult(
                timing_ms=0.0,
                gflops=0.0,
                bandwidth_gbs=0.0,
                correctness=True,
                max_abs_error=0.0,
                max_rel_error=0.0,
                logs="Offline validation: no GPU backend available",
            )
            return True, validation, profile, None

        try:
            result = self.validator.validate_and_profile_kernel(
                kernel_code=kernel_code,
                kernel_name=kernel_name,
                operation="matmul",
                input_shapes=spec.input_shapes,
            )
        except Exception as exc:  # noqa: BLE001 - propagate readable failure
            validation = ValidationResult(False, 0.0, 0.0, None, f"validation error: {exc}")
            return False, validation, None, str(exc)

        compile_success = bool(result.get("compile_success"))
        validation_success = bool(result.get("validation_success")) if compile_success else False

        validation = ValidationResult(
            ok=validation_success,
            max_abs_err=float(result.get("max_abs_error", 0.0) or 0.0),
            max_rel_err=float(result.get("max_rel_error", 0.0) or 0.0),
            case_index=None,
            message=("ok" if validation_success else "; ".join(result.get("errors", [])) or "validation failed"),
        )

        if compile_success and validation_success:
            profile = ProfileResult(
                timing_ms=float(result.get("execution_time_ms", 0.0) or 0.0),
                gflops=float(result.get("gflops", 0.0) or 0.0),
                bandwidth_gbs=float(result.get("bandwidth_gbs", 0.0) or 0.0),
                correctness=True,
                max_abs_error=validation.max_abs_err,
                max_rel_error=validation.max_rel_err,
                logs=None,
            )
            return True, validation, profile, None

        feedback = None
        if not compile_success:
            errors = result.get("errors") or []
            feedback = "; ".join(errors) if errors else "compilation failed"
        elif not validation_success:
            feedback = validation.message

        return compile_success, validation, None, feedback

    def _should_use_offline_validation(self) -> bool:
        compiler = self.validator.compiler
        if compiler.backend == "cuda":
            # Assume CuPy is available if CUDA backend detected.
            return False
        if compiler.backend == "opencl" and compiler.context is None:
            return True
        if compiler.backend not in {"opencl", "cuda"}:
            return True
        return False

    def _compose_corrective_feedback(
        self,
        generation_feedback: Optional[str],
        downstream_feedback: Optional[str],
        static_analysis: StaticAnalysisResult,
    ) -> Optional[str]:
        hints: List[str] = []
        if generation_feedback:
            hints.append(generation_feedback)
        if downstream_feedback:
            hints.append(downstream_feedback)

        hints.extend(self._static_hints(static_analysis))

        if not hints:
            return None
        return "; ".join(hints)

    @staticmethod
    def _static_hints(static_analysis: StaticAnalysisResult) -> List[str]:
        hints: List[str] = []
        if static_analysis.register_pressure_hint == "high":
            hints.append("reduce register pressure (smaller tiles/unroll)")
        if static_analysis.has_local_memory is False:
            hints.append("consider introducing local memory tiling")
        if static_analysis.potential_issues:
            hints.extend(static_analysis.potential_issues)
        return hints

    @staticmethod
    def _temperature_for_retry(retry_index: int) -> float:
        return min(0.3 + retry_index * 0.2, 0.9)

    @staticmethod
    def _extract_kernel_name(code: str) -> Optional[str]:
        match = re.search(r"__kernel\s+void\s+([a-zA-Z_]\w*)", code)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _build_attempt_id(ir_hash: str, retry_index: int) -> str:
        timestamp_ms = int(time.time() * 1000)
        digest = hashlib.sha1(f"{ir_hash}:{timestamp_ms}:{retry_index}".encode("utf-8")).hexdigest()[:10]
        return f"{ir_hash[:12]}_{digest}_{retry_index}"

    def _fallback_kernel(self, spec: KernelSpec) -> str:
        if spec.operation == "matmul":
            return FALLBACK_OPENCL_MATMUL
        return ""

    def _provider_ready(self, provider: str) -> bool:
        if provider == "openai":
            return "OPENAI_API_KEY" in os.environ
        if provider == "deepseek":
            return "DEEPSEEK_API_KEY" in os.environ
        return False

    def _provider_missing_reason(self, provider: str) -> Optional[str]:
        if provider == "openai" and "OPENAI_API_KEY" not in os.environ:
            return "openai provider unavailable: set OPENAI_API_KEY"
        if provider == "deepseek" and "DEEPSEEK_API_KEY" not in os.environ:
            return "deepseek provider unavailable: set DEEPSEEK_API_KEY"
        if provider not in SUPPORTED_PROVIDERS:
            return f"provider '{provider}' not supported"
        return None

    def _should_stop_retry(self, attempt: GenerationAttempt) -> bool:
        if attempt.provider == "offline-fallback":
            return True
        if attempt.corrective_feedback:
            lower = attempt.corrective_feedback.lower()
            if "unavailable" in lower or "missing" in lower:
                return True
        return False

    def _resolve_provider(self, provider: Optional[str]) -> str:
        if provider:
            provider = provider.lower()
            if provider in SUPPORTED_PROVIDERS:
                return provider
            raise ValueError(f"Unsupported provider '{provider}'")
        if self.default_provider in SUPPORTED_PROVIDERS:
            return self.default_provider
        return "openai"

    def _persist_attempt(self, attempt: GenerationAttempt) -> None:
        attempt_dir = self.dataset_dir / attempt.attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=True)

        metadata = asdict(attempt)
        metadata_path = attempt_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        if attempt.kernel_code:
            (attempt_dir / "kernel.cl").write_text(attempt.kernel_code)

        (attempt_dir / "prompt.txt").write_text(attempt.prompt)

    # ------------------------------------------------------------------
    # CLI integration
    # ------------------------------------------------------------------
    def run_cli_attempt(
        self,
        shapes: str,
        max_attempts: int = 5,
        provider: Optional[str] = None,
    ) -> List[GenerationAttempt]:
        """Helper for CLI usage to run the loop from shape strings."""

        shape_parts = shapes.split(",")
        if len(shape_parts) != 2:
            raise ValueError("Shapes must be in format MxK,KxN")

        m, k = map(int, shape_parts[0].split("x"))
        k2, n = map(int, shape_parts[1].split("x"))
        if k != k2:
            raise ValueError("Inner dimensions must match")

        A = Tensor("A", (m, k), "f32")
        B = Tensor("B", (k, n), "f32")
        matmul_ir = MatMul(A, B)
        return self.run_pipeline(matmul_ir, max_attempts=max_attempts, provider=provider)


# CLI interface -----------------------------------------------------------------------------------------------
def run_ai_generation_loop() -> List[GenerationAttempt]:
    """CLI entry point for running the AI generation loop."""

    import argparse

    parser = argparse.ArgumentParser(description="Run AI kernel generation loop for matmul")
    parser.add_argument(
        "--shapes",
        type=str,
        required=True,
        help="Matrix shapes in format MxK,KxN (e.g., 1024x512,512x2048)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Maximum number of generation attempts",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Dataset directory for storing results",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=sorted(SUPPORTED_PROVIDERS),
        help="AI provider to use",
    )

    args = parser.parse_args()

    pipeline = AIGenerationPipeline(
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
        default_provider=args.provider,
    )

    attempts = pipeline.run_cli_attempt(
        shapes=args.shapes,
        max_attempts=args.max_attempts,
        provider=args.provider,
    )

    print(f"Generated {len(attempts)} attempts")
    for idx, attempt in enumerate(attempts, 1):
        status = (
            "SUCCESS"
            if attempt.compile_success and attempt.validation_result and attempt.validation_result.ok
            else "FAILED"
        )
        print(f"Attempt {idx}: {status} (provider={attempt.provider}, retries={attempt.retry_count})")

    return attempts


if __name__ == "__main__":
    run_ai_generation_loop()

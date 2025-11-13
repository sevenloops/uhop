"""End-to-end tests for the AI generation pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import pytest

# Ensure modules resolve when running from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from uhop.ai_codegen.pipeline import (  # noqa: E402
    AIGenerationPipeline,
    StaticAnalyzer,
)
from uhop.ir import MatMul, Tensor  # noqa: E402


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Provide an isolated dataset directory for the tests."""

    return tmp_path_factory.mktemp("ai_pipeline_dataset")


@pytest.fixture(scope="module")
def pipeline(dataset_dir: Path) -> AIGenerationPipeline:
    """Instantiate the pipeline with a temporary dataset location."""

    return AIGenerationPipeline(dataset_dir=dataset_dir)


@pytest.fixture(scope="module")
def matmul_ir() -> MatMul:
    """Create a compact MatMul IR for quick test cycles."""

    A = Tensor("A", (32, 64), "f32")
    B = Tensor("B", (64, 32), "f32")
    return MatMul(A, B)


@pytest.fixture(scope="module")
def pipeline_attempts(
    pipeline: AIGenerationPipeline, matmul_ir: MatMul
) -> List:
    """Run the pipeline once and memoize the attempts for reuse."""

    attempts = pipeline.run_pipeline(matmul_ir, max_attempts=2)
    assert attempts, "pipeline should produce at least one attempt"
    return attempts


def test_static_analysis_detects_resources() -> None:
    """Static analyzer should flag local memory usage and barriers."""

    sample_kernel = """
    __kernel void analyzed_kernel(const int M, const int N, const int K,
                                 __global const float* A,
                                 __global const float* B,
                                 __global float* C) {
        __local float tileA[16][16];
        __local float tileB[16][16];

        int row = get_global_id(0);
        int col = get_global_id(1);
        int local_row = get_local_id(0);
        int local_col = get_local_id(1);

        float sum = 0.0f;
        for (int k = 0; k < K; k += 16) {
            if (row < M && k + local_col < K) {
                tileA[local_row][local_col] = A[row * K + k + local_col];
            }
            if (k + local_row < K && col < N) {
                tileB[local_row][local_col] = B[(k + local_row) * N + col];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int i = 0; i < 16; i++) {
                sum += tileA[local_row][i] * tileB[i][local_col];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (row < M && col < N) {
            C[row * N + col] = sum;
        }
    }
    """

    analyzer = StaticAnalyzer()
    analysis = analyzer.analyze_opencl_kernel(sample_kernel)

    assert analysis.has_local_memory is True
    assert analysis.has_barriers is True
    assert "indexed_access" in analysis.memory_access_patterns


def test_pipeline_attempt_contains_validation(pipeline_attempts) -> None:
    """Generated attempts should include validation results even offline."""

    latest = pipeline_attempts[-1]
    assert latest.compile_success is True
    assert latest.validation_result is not None
    assert latest.validation_result.ok is True
    assert latest.profile_result is not None
    assert latest.profile_result.correctness is True


def test_attempt_feedback_is_string(pipeline_attempts) -> None:
    """Corrective feedback should be a descriptive string when present."""

    feedback_values = {
        attempt.corrective_feedback
        for attempt in pipeline_attempts
        if attempt.corrective_feedback
    }
    for feedback in feedback_values:
        assert isinstance(feedback, str)
        assert feedback  # non-empty


def test_dataset_persistence(dataset_dir: Path, pipeline_attempts) -> None:
    """Each attempt should result in a persisted metadata file."""

    recorded_ids = {attempt.attempt_id for attempt in pipeline_attempts}

    for attempt_id in recorded_ids:
        attempt_path = dataset_dir / attempt_id
        metadata_path = attempt_path / "metadata.json"
        kernel_path = attempt_path / "kernel.cl"
        prompt_path = attempt_path / "prompt.txt"

        assert metadata_path.exists()
        assert kernel_path.exists()
        assert prompt_path.exists()

        metadata = json.loads(metadata_path.read_text())
        assert metadata["attempt_id"] == attempt_id
        assert metadata["compile_success"] is True


def test_dataset_query_counts(dataset_dir: Path, pipeline_attempts) -> None:
    """Dataset metadata should be queryable and reflect validation status."""

    attempts = []
    for attempt_dir in dataset_dir.iterdir():
        metadata_file = attempt_dir / "metadata.json"
        if metadata_file.exists():
            attempts.append(json.loads(metadata_file.read_text()))

    assert attempts, "persisted dataset should contain entries"

    successful = [
        a
        for a in attempts
        if a.get("compile_success") and a.get("validation_result", {}).get("ok")
    ]
    assert len(successful) == len(attempts)

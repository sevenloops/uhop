# AI Generation Pipeline

This document describes the closed-loop AI kernel generation system implemented in UHOP.

## Overview

The AI Generation Pipeline implements a complete closed-loop system for generating, validating, and optimizing GPU kernels using AI models. The pipeline follows this workflow:

1. **Spec Extraction** - Extract kernel specifications from IR + hardware fingerprint
2. **AI Prompt** - Compose prompts including prior tuned parameters and constraints
3. **Static Analysis** - Analyze generated kernels for resource usage patterns
4. **Compilation** - Compile generated kernels
5. **Micro-testing** - Validate correctness against reference implementations
6. **Profiling** - Measure performance (timing, GFLOPS, bandwidth)
7. **Feedback Loop** - Provide corrective prompting for failures/slow kernels
8. **Archiving** - Persist full metadata and artifacts

## Architecture

### Core Components

- **`AIGenerationPipeline`** - Main pipeline orchestrator
- **`KernelSpec`** - Extracted specification from IR + hardware
- **`StaticAnalyzer`** - Resource usage analysis
- **`GenerationAttempt`** - Full record of generation attempts
- **Multi-provider abstraction** - Support for OpenAI and DeepSeek

### Provider Support

#### OpenAI

- Models: GPT-4, GPT-4o-mini, etc.
- Environment variable: `OPENAI_API_KEY`
- Configurable via `UHOP_OPENAI_MODEL`

#### DeepSeek

- Models: deepseek-coder, etc.
- Environment variable: `DEEPSEEK_API_KEY`
- Configurable via `UHOP_DEEPSEEK_MODEL`

## Usage

### CLI Commands

```bash
# Run AI generation for matmul
python -m uhop.cli_ai run --shapes 1024x512,512x2048

# List all generation attempts
python -m uhop.cli_ai list

# Show details of specific attempt
python -m uhop.cli_ai show <attempt_id>

# Query dataset by device or operation
python -m uhop.cli_ai query --device nvidia --operation matmul --success-only
```

### Python API

```python
from uhop.ai_codegen.pipeline import AIGenerationPipeline
from uhop.ir.ir import MatMul, Tensor

# Create matmul IR
A = Tensor("A", (128, 256), "f32")
B = Tensor("B", (256, 64), "f32")
matmul_ir = MatMul(A, B)

# Initialize pipeline
pipeline = AIGenerationPipeline()

# Run full pipeline
attempts = pipeline.run_pipeline(matmul_ir, max_attempts=5)

# Generate single kernel
spec = pipeline.extract_spec_from_ir(matmul_ir)
attempt = pipeline.generate_kernel(spec, provider="openai")
```

### Demo Script

```bash
python examples/ai_generation_demo.py
```

## Configuration

### Environment Variables

| Variable              | Description        | Default          |
| --------------------- | ------------------ | ---------------- |
| `OPENAI_API_KEY`      | OpenAI API key     | Required         |
| `DEEPSEEK_API_KEY`    | DeepSeek API key   | Required         |
| `UHOP_OPENAI_MODEL`   | OpenAI model       | `gpt-4o-mini`    |
| `UHOP_DEEPSEEK_MODEL` | DeepSeek model     | `deepseek-coder` |
| `UHOP_AI_PROVIDER`    | Default provider   | `openai`         |
| `UHOP_AI_MAX_RETRIES` | Max retry attempts | `3`              |
| `UHOP_AI_DEBUG`       | Debug logging      | `0`              |

### Dataset Structure

Generated attempts are stored in `~/.uhop_ai_dataset/` with this structure:

```text
.uhop_ai_dataset/
├── <attempt_id>/
│   ├── metadata.json     # Full generation metadata
│   ├── kernel.cl         # Generated kernel code
│   └── prompt.txt        # AI prompt used
```

## Static Analysis

The static analyzer examines generated kernels for:

- **Local memory usage** - Detection of `__local` memory
- **Barrier synchronization** - Presence of `barrier()` calls
- **Workgroup size hints** - Analysis of `get_local_size()` usage
- **Register pressure** - Estimation based on variable count
- **Memory access patterns** - Vector loads/stores, indexed access
- **Potential issues** - Conditional barriers, barriers in loops

## Prompt Engineering

Prompts are composed using templates from `uhop/ai_codegen/prompt_templates.py`:

- **Operation-specific prompts** for matmul, relu, conv2d
- **Architecture-specific prompts** for CUDA, OpenCL, HIP, Metal, Triton
- **Constraint integration** from IR schedules
- **Corrective feedback** from prior failed attempts

## Implementation Status

### Completed

- [x] Spec extraction from IR + hardware fingerprint
- [x] Multi-provider abstraction (OpenAI, DeepSeek)
- [x] Prompt composer with constraints and prior attempts
- [x] Static analyzer for resource hints
- [x] Dataset persistence and querying
- [x] CLI interface for running pipeline
- [x] Retry logic with corrective prompting

### In Progress

- [ ] Kernel compilation and validation
- [ ] Performance profiling (timing, GFLOPS, bandwidth)
- [ ] Integration with existing backend systems
- [ ] Advanced corrective feedback mechanisms

### Future Enhancements

- Support for more operations (conv2d, relu, fused ops)
- Integration with autotuning system
- Advanced static analysis for performance prediction
- Multi-objective optimization (performance vs. correctness)
- Distributed generation across multiple providers

## Testing

Run the test suite:

```bash
# Run demo script
python examples/ai_generation_demo.py

# Test CLI commands
python -m uhop.cli_ai list
python -m uhop.cli_ai query --operation matmul
```

## Troubleshooting

### Common Issues

1. **API Key Missing**

   - Set `OPENAI_API_KEY` or `DEEPSEEK_API_KEY` environment variables

2. **Generation Failures**

   - Check API rate limits and quotas
   - Verify model names are correct
   - Enable debug logging with `UHOP_AI_DEBUG=1`

3. **Compilation Failures**
   - Generated kernels may require manual fixes
   - Static analysis provides hints for common issues

### Debugging

Enable verbose logging:

```bash
export UHOP_AI_DEBUG=1
python examples/ai_generation_demo.py
```

## Contributing

When adding new operations:

1. Add operation-specific prompts to `prompt_templates.py`
2. Update static analyzer for new patterns
3. Add IR extraction support
4. Update CLI with new operation options

## Related Documentation

- [Core IR System](IR_ENHANCED.md)
- [Validation Harness](VALIDATION.md)
- [Autotuning System](AUTOTUNE.md)
- [Backend Architecture](BACKENDS.md)

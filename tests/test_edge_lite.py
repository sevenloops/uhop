#!/usr/bin/env python3
"""
Test UHOP Edge Lite Backend

This tests the edge lite mode which uses only numpy + OpenCL,
without PyTorch dependency. Perfect for edge devices like:
- Raspberry Pi
- Jetson Nano/Orin
- Mobile devices with ARM GPUs
- IoT devices

Run with: python3 test_edge_lite.py
"""

import sys
import time
import numpy as np

print("=" * 60)
print("UHOP Edge Lite Backend Test")
print("=" * 60)
print()

# Test 1: Import lite backend
print("Test 1: Import lite backend...")
try:
    from uhop.backends.lite_backend import (
        is_lite_backend_available,
        is_lite_opencl_available,
        lite_matmul,
        lite_conv2d,
        lite_relu,
        get_edge_device_info,
        print_edge_device_info,
    )
    print("✓ Successfully imported lite backend")
except Exception as e:
    print(f"✗ Failed to import lite backend: {e}")
    sys.exit(1)

print()

# Test 2: Check availability
print("Test 2: Check backend availability...")
print(f"  Lite backend available: {is_lite_backend_available()}")
print(f"  OpenCL GPU available: {is_lite_opencl_available()}")
print()

# Test 3: Display device info
print("Test 3: Display device information...")
print_edge_device_info()
print()

# Test 4: Test matmul (CPU)
print("Test 4: Matrix multiplication (CPU fallback)...")
try:
    A = np.random.randn(64, 64).astype(np.float32)
    B = np.random.randn(64, 64).astype(np.float32)

    start = time.perf_counter()
    C = lite_matmul(A, B, use_gpu=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Verify correctness
    expected = np.matmul(A, B)
    error = np.max(np.abs(C - expected))

    print(f"  Shape: {C.shape}")
    print(f"  Time: {elapsed_ms:.2f} ms")
    print(f"  Max error: {error:.2e}")

    if error < 1e-4:
        print("  ✓ Correctness verified")
    else:
        print("  ✗ Correctness check failed!")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print()

# Test 5: Test matmul (GPU if available)
print("Test 5: Matrix multiplication (GPU if available)...")
try:
    A = np.random.randn(512, 512).astype(np.float32)
    B = np.random.randn(512, 512).astype(np.float32)

    start = time.perf_counter()
    C = lite_matmul(A, B, use_gpu=True)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Calculate GFLOPS
    flops = 2 * 512 * 512 * 512
    gflops = (flops / (elapsed_ms / 1000)) / 1e9

    # Verify correctness
    expected = np.matmul(A, B)
    error = np.max(np.abs(C - expected))

    print(f"  Shape: {C.shape}")
    print(f"  Time: {elapsed_ms:.2f} ms")
    print(f"  Performance: {gflops:.1f} GFLOPS")
    print(f"  Max error: {error:.2e}")

    if error < 1e-3:
        print("  ✓ Correctness verified")
    else:
        print("  ✗ Correctness check failed!")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print()

# Test 6: Test ReLU
print("Test 6: ReLU activation...")
try:
    X = np.random.randn(1000, 1000).astype(np.float32)

    start = time.perf_counter()
    Y = lite_relu(X, use_gpu=True)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Verify correctness
    expected = np.maximum(0, X)
    error = np.max(np.abs(Y - expected))

    print(f"  Shape: {Y.shape}")
    print(f"  Time: {elapsed_ms:.2f} ms")
    print(f"  Max error: {error:.2e}")

    if error < 1e-6:
        print("  ✓ Correctness verified")
    else:
        print("  ✗ Correctness check failed!")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print()

# Test 7: Test Conv2D (small)
print("Test 7: 2D Convolution (small)...")
try:
    # Small conv: N=1, C_in=3, H=W=8, C_out=8, K=3
    input_data = np.random.randn(1, 3, 8, 8).astype(np.float32)
    weight = np.random.randn(8, 3, 3, 3).astype(np.float32)

    start = time.perf_counter()
    output = lite_conv2d(input_data, weight, stride=1, padding=1, use_gpu=True)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"  Input: {input_data.shape}")
    print(f"  Weight: {weight.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Time: {elapsed_ms:.2f} ms")
    print("  ✓ Conv2D completed")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print()

# Test 8: Benchmark comparison
print("Test 8: Performance comparison (1024x1024 matmul)...")
try:
    size = 1024
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)

    # CPU numpy baseline
    start = time.perf_counter()
    C_cpu = np.matmul(A, B)
    time_cpu = (time.perf_counter() - start) * 1000

    # Lite backend (GPU if available)
    start = time.perf_counter()
    C_gpu = lite_matmul(A, B, use_gpu=True)
    time_gpu = (time.perf_counter() - start) * 1000

    # Calculate metrics
    flops = 2 * size * size * size
    gflops_cpu = (flops / (time_cpu / 1000)) / 1e9
    gflops_gpu = (flops / (time_gpu / 1000)) / 1e9
    speedup = time_cpu / time_gpu

    print(f"  CPU (NumPy):  {time_cpu:.1f} ms  ({gflops_cpu:.1f} GFLOPS)")
    print(f"  GPU (Lite):   {time_gpu:.1f} ms  ({gflops_gpu:.1f} GFLOPS)")
    print(f"  Speedup:      {speedup:.2f}x")

    error = np.max(np.abs(C_gpu - C_cpu))
    print(f"  Max error:    {error:.2e}")

    if speedup > 1.0:
        print("  ✓ GPU is faster than CPU")
    else:
        print("  ℹ GPU not faster (may be using CPU fallback)")

except Exception as e:
    print(f"  ✗ Failed: {e}")

print()
print("=" * 60)
print("Edge Lite Backend Test Complete")
print("=" * 60)
print()

# Print recommendations
device_info = get_edge_device_info()
if device_info['has_gpu']:
    print("✓ Your system has OpenCL GPU support!")
    print("  The lite backend can use GPU acceleration.")
    print()
    print("  Recommended tile sizes for your hardware:")
    from uhop.backends.lite_backend import get_edge_optimization_hints
    hints = get_edge_optimization_hints()
    print(f"    - Tile size: {hints['tile_size']}")
    print(f"    - Use FP16: {hints['use_fp16']}")
    print(f"    - Memory efficient: {hints['memory_efficient']}")
else:
    print("ℹ No OpenCL GPU detected.")
    print("  The lite backend will use CPU (NumPy) fallback.")
    print()
    print("  To enable GPU acceleration:")
    print("    - Install OpenCL drivers for your hardware")
    print("    - pip install pyopencl")

print()
print("Next steps:")
print("  1. Test on actual edge device (Jetson, Raspberry Pi)")
print("  2. Tune tile sizes for specific hardware")
print("  3. Add FP16 kernels for mobile GPUs")
print("  4. Implement power/thermal awareness")

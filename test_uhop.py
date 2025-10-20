#!/usr/bin/env python3
"""
Test script to verify UHOP functionality with OpenCL.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_hardware_detection():
    """Test hardware detection."""
    print("=== Testing Hardware Detection ===")
    from uhop.hardware import detect_hardware
    
    hw = detect_hardware()
    print(f"Vendor: {hw.vendor}")
    print(f"Kind: {hw.kind}")
    print(f"Name: {hw.name}")
    print(f"Details: {hw.details}")
    
    return hw

def test_backends():
    """Test backend availability."""
    print("\n=== Testing Backend Availability ===")
    from uhop.backends import (
        is_torch_available,
        is_triton_available,
        is_opencl_available,
    )
    
    print(f"Torch available: {is_torch_available()}")
    print(f"Triton available: {is_triton_available()}")
    print(f"OpenCL available: {is_opencl_available()}")

def test_demo():
    """Test a simple demo."""
    print("\n=== Testing UHOP Demo ===")
    try:
        from uhop.web_api import _demo_matmul
        
        print("Running 64x64 matrix multiplication demo...")
        result = _demo_matmul(64, 2)
        
        print("Demo completed successfully!")
        print(f"UHOP time: {result['timings']['uhop']:.6f}s")
        print(f"Naive time: {result['timings']['naive']:.6f}s")
        print(f"UHOP wins: {result['timings']['uhop_won']}")
        
        return True
    except Exception as e:
        print(f"Demo failed: {e}")
        return False

if __name__ == "__main__":
    print("UHOP Functionality Test")
    print("=" * 50)
    
    hw = test_hardware_detection()
    test_backends()
    success = test_demo()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! UHOP is working correctly.")
        if hw.kind == 'opencl':
            print("🚀 OpenCL GPU acceleration is available!")
    else:
        print("❌ Some tests failed.")
    
    print("\nYou can now:")
    print("1. Start the backend: cd backend && node index.js")
    print("2. Start the frontend: cd frontend && npm run dev") 
    print("3. Open http://localhost:5173 in your browser")
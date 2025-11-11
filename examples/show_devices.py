# examples/show_devices.py
import sys

print("Python:", sys.version)

try:
    import torch

    print("Torch available:", True)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    mps = getattr(torch.backends, "mps", None)
    print("MPS available:", bool(mps and torch.backends.mps.is_available()))
    # UHOP torch device preference
    try:
        from uhop.backends.torch_backend import _torch_device_preference

        dev = _torch_device_preference()
        pref = getattr(dev, "type", None) if dev else None
        print("UHOP preferred device:", pref)
    except Exception:
        pass
except Exception as e:
    print("Torch available:", False, e)

try:
    import pyopencl as cl

    plats = cl.get_platforms()
    print("OpenCL platforms:", len(plats))
    for pi, p in enumerate(plats):
        print(f"  Platform {pi}:", p.name)
        for di, d in enumerate(p.get_devices()):
            if d.type & cl.device_type.GPU:
                kind = "GPU"
            elif d.type & cl.device_type.CPU:
                kind = "CPU"
            else:
                kind = str(d.type)
            print(f"    Device {di}: {d.name} [{kind}]")
    # Build context using our backend policy to see selection
    from uhop.backends.opencl_backend import _OPENCL_AVAILABLE

    print("pyopencl available via backend:", _OPENCL_AVAILABLE)
    from uhop.backends.opencl_backend import is_opencl_available

    print("is_opencl_available():", is_opencl_available())
except Exception as e:
    print("OpenCL error:", e)

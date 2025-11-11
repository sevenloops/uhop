"""
Minimal OpenCL wrapper using PyOpenCL for compiling & launching kernels.
"""

try:
    import pyopencl as cl  # type: ignore
except Exception:
    cl = None  # type: ignore


def ensure_opencl():
    if cl is None:
        raise RuntimeError("pyopencl is required for the OpenCL backend. Install pyopencl.")


class OpenCLKernel:
    def __init__(self, source: str, kernel_name: str):
        ensure_opencl()
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        self.program = cl.Program(self.ctx, source).build()
        self.kernel = getattr(self.program, kernel_name)

    def launch(self, global_size, local_size, args):
        # Prepare device buffers and set kernel args. Caller must pass device buffers (cl.Buffer) or numpy arrays.
        cl_args = []
        for a in args:
            if isinstance(a, cl.Buffer):
                cl_args.append(a)
            else:
                # assume numpy array; transfer to device buffer
                mf = cl.mem_flags
                buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
                cl_args.append(buf)
        # set kernel args automatically
        self.kernel.set_args(*cl_args)
        evt = cl.enqueue_nd_range_kernel(self.queue, self.kernel, global_size, local_size)
        evt.wait()
        return evt


def time_kernel_run(kernel: OpenCLKernel, global_size, local_size, args, warmups=2, runs=6):
    ensure_opencl()
    # Warmups
    for _ in range(warmups):
        kernel.launch(global_size, local_size, args)
    # timed runs using profiling info
    durations = []
    for _ in range(runs):
        evt = kernel.launch(global_size, local_size, args)
        # profiling info available
        start = evt.profile.start
        end = evt.profile.end
        durations.append((end - start) * 1e-9)
    # return average seconds
    return sum(durations) / len(durations)


def arrays_to_device(ctx, queue, *arrays, dtype=None):
    ensure_opencl()
    mf = cl.mem_flags
    devbufs = []
    for a in arrays:
        import numpy as _np  # local import to avoid global dependency if not used

        if isinstance(a, _np.ndarray):
            devbufs.append(
                cl.Buffer(
                    ctx,
                    mf.READ_ONLY | mf.COPY_HOST_PTR,
                    hostbuf=a.astype(dtype or a.dtype),
                )
            )
        else:
            raise TypeError("arrays_to_device expects numpy arrays")
    return devbufs

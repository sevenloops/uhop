# uhop/pytorch_wrappers.py
"""
PyTorch autograd wrappers so UHOP can be used inside training loops.

UHOPConv2DFunction.forward:
  - Calls UHOP optimizer-dispatched conv2d (which may run on torch/triton/opencl/AI)
  - Saves tensors for backward.

UHOPConv2DFunction.backward:
  - For Phase1 we use a CPU torch fallback to compute gradients (correctness-first).
  - Later we will implement backend native backward kernels (OpenCL/HIP/Triton).
"""
import torch
from torch.autograd import Function
import numpy as np
from .optimizer import _GLOBAL_OPT

class UHOPConv2DFunction(Function):
    @staticmethod
    def forward(ctx, input_t, weight_t, stride=1, padding=0):
        # convert to numpy and call UHOP conv2d implementation
        inp_np = input_t.detach().cpu().numpy()
        w_np = weight_t.detach().cpu().numpy()
        # UHOP decorator expects a function, but we can call optimizer logic directly:
        # use the cached or selected backend via the optimize decorator on a small wrapper
        @(_GLOBAL_OPT.optimize("conv2d"))
        def conv_np(a, b, stride=1, padding=0):
            # baseline numpy conv (naive)
            import numpy as _np
            N, C, H, W = a.shape
            Cout, Cin, KH, KW = b.shape
            outH = H - KH + 1
            outW = W - KW + 1
            out = _np.zeros((N, Cout, outH, outW), dtype=_np.float32)
            for n in range(N):
                for co in range(Cout):
                    for y in range(outH):
                        for x in range(outW):
                            s = 0.0
                            for ci in range(Cin):
                                for ky in range(KH):
                                    for kx in range(KW):
                                        s += a[n,ci,y+ky,x+kx] * b[co,ci,ky,kx]
                            out[n,co,y,x] = s
            return out

        out_np = conv_np(inp_np, w_np, stride=stride, padding=padding)
        out_t = torch.from_numpy(out_np).to(input_t.device)
        ctx.save_for_backward(input_t, weight_t)
        ctx.stride = stride
        ctx.padding = padding
        return out_t

    @staticmethod
    def backward(ctx, grad_output):
        input_t, weight_t = ctx.saved_tensors
        # Fallback: compute gradients using torch CPU autograd for correctness (inefficient).
        import torch.nn.functional as F
        input_cpu = input_t.detach().cpu().requires_grad_(True)
        weight_cpu = weight_t.detach().cpu().requires_grad_(True)
        out = F.conv2d(input_cpu, weight_cpu, stride=ctx.stride, padding=ctx.padding)
        out.backward(grad_output.detach().cpu())
        grad_input = input_cpu.grad.to(grad_output.device)
        grad_weight = weight_cpu.grad.to(grad_output.device)
        return grad_input, grad_weight, None, None

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


class UHOPConv2DFunction(Function):
    @staticmethod
    def forward(ctx, input_t, weight_t, stride=1, padding=0):
        # Fast path: use torch conv2d directly (no autograd graph is recorded here).
        try:
            import torch.nn.functional as F

            out_t = F.conv2d(input_t, weight_t, stride=stride, padding=padding)
        except Exception:
            # Fallback: very slow naive NumPy conv
            inp_np = input_t.detach().cpu().numpy()
            w_np = weight_t.detach().cpu().numpy()
            import numpy as _np

            N, C, H, W = inp_np.shape
            Cout, Cin, KH, KW = w_np.shape
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
                                        s += (
                                            inp_np[n, ci, y + ky, x + kx]
                                            * w_np[co, ci, ky, kx]
                                        )
                            out[n, co, y, x] = s
            out_t = torch.from_numpy(out).to(input_t.device)
        ctx.save_for_backward(input_t, weight_t)
        ctx.stride = stride
        ctx.padding = padding
        return out_t

    @staticmethod
    def backward(ctx, grad_output):
        input_t, weight_t = ctx.saved_tensors
        # Fallback: compute gradients using torch CPU autograd for correctness (inefficient).
        from torch.nn.grad import conv2d_input, conv2d_weight

        input_cpu = input_t.detach().cpu()
        weight_cpu = weight_t.detach().cpu()
        grad_out_cpu = grad_output.detach().cpu()
        # Compute grads directly using torch.nn.grad helpers (no new graph built)
        grad_input = conv2d_input(
            input_cpu.shape,
            weight_cpu,
            grad_out_cpu,
            stride=ctx.stride,
            padding=ctx.padding,
        )
        grad_weight = conv2d_weight(
            input_cpu,
            weight_cpu.shape,
            grad_out_cpu,
            stride=ctx.stride,
            padding=ctx.padding,
        )
        grad_input = grad_input.to(grad_output.device)
        grad_weight = grad_weight.to(grad_output.device)
        return grad_input, grad_weight, None, None

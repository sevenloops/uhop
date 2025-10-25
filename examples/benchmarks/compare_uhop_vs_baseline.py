# examples/compare_uhop_vs_baseline.py
"""
Run UHOP-wrapped conv2d vs baseline Conv2d with identical weights/inputs.
Report loss and median step time for fair comparison.
"""
import time
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from uhop.pytorch_wrappers import UHOPConv2DFunction

class UHopCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(8, 3, 3, 3))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
    def forward(self, x):
        return UHOPConv2DFunction.apply(x, self.weight, 1, 0)

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, stride=1, padding=0, bias=False)
    def forward(self, x):
        return self.conv(x)


def run_model(model, x, y, steps=20):
    opt = optim.SGD(model.parameters(), lr=1e-2)
    times = []
    last_loss = None
    for _ in range(steps):
        t0 = time.perf_counter()
        pred = model(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        last_loss = float(loss.item())
    return last_loss, statistics.median(times)


def main():
    torch.manual_seed(0)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randn(8, 8, 30, 30)

    # Start from identical weights
    base = BaselineCNN()
    uhop = UHopCNN()
    with torch.no_grad():
        uhop.weight.copy_(base.conv.weight)

    bl_loss, bl_med = run_model(base, x, y)
    uh_loss, uh_med = run_model(uhop, x, y)

    print("Baseline: loss=%.6f, median_step=%.4f s" % (bl_loss, bl_med))
    print("UHOP    : loss=%.6f, median_step=%.4f s" % (uh_loss, uh_med))

if __name__ == "__main__":
    main()

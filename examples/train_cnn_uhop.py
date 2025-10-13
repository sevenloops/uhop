# examples/train_cnn_uhop.py
"""
Tiny CNN training demo using UHOP-optimized conv2d + relu.
This demo uses the UHOP autograd wrapper for conv2d (forward uses UHOP dispatch,
backward uses a CPU torch fallback for correctness in Phase1).
"""
import torch
import math
import torch.nn as nn
import torch.optim as optim
from uhop.pytorch_wrappers import UHOPConv2DFunction
from uhop import UHopOptimizer

hop = UHopOptimizer()

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Using a parameterized conv weight so we can update it
        self.weight = nn.Parameter(torch.empty(8, 3, 3, 3))  # Cout, Cin, KH, KW
        # Initialize like nn.Conv2d default (Kaiming uniform)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # x: (N, C, H, W)
        return UHOPConv2DFunction.apply(x, self.weight, 1, 0)

def train_one_epoch():
    torch.manual_seed(0)
    model = TinyCNN()
    opt = optim.SGD(model.parameters(), lr=1e-2)
    for i in range(10):
        x = torch.randn(4, 3, 16, 16)
        y = torch.randn(4, 8, 14, 14)
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {i} loss {loss.item():.6f}")

if __name__ == "__main__":
    print("[UHOP] Starting tiny training demo")
    train_one_epoch()
    print("[UHOP] Demo finished")

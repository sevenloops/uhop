# examples/train_cnn_baseline.py
"""
Baseline tiny CNN training using native torch.nn.Conv2d (no UHOP wrapper).
Runs a few steps to compare behavior and loss values against UHOP demo.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=0, bias=False)

    def forward(self, x):
        return self.conv(x)


def train_one_epoch():
    torch.manual_seed(0)
    model = TinyCNN()
    opt = optim.SGD(model.parameters(), lr=1e-2)
    for i in range(10):
        x = torch.randn(4, 3, 16, 16)
        y = torch.randn(4, 8, 14, 14)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"[baseline] step {i} loss {loss.item():.6f}")

if __name__ == "__main__":
    print("[BASELINE] Starting tiny training demo (no UHOP)")
    train_one_epoch()
    print("[BASELINE] Demo finished")

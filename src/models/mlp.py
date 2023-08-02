from turtle import forward
import torch.nn as nn
import torch
from torchsummary import summary

class NodeFeatureEncoder(nn.Module):
    def __init__(self, cfg, ckpt=None, in_dim=None):
        super(NodeFeatureEncoder, self).__init__()
        if in_dim is None:
            in_dim = 516 if cfg.SOLVER.TYPE == 'TG' else 512

        self.layer = nn.Sequential(
            nn.Linear(in_dim, 128), # 516
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.to(cfg.MODEL.DEVICE)
        if ckpt is not None:
            self.load_state_dict(ckpt)

    def forward(self, x):
        return self.layer(x)

class EdgeFeatureEncoder(nn.Module):
    def __init__(self, cfg, ckpt=None, in_dim=None):
        super(EdgeFeatureEncoder, self).__init__()
        if in_dim is None:
            # 6: add velocity feature
            in_dim = 6 if cfg.SOLVER.TYPE == 'TG' else 4
        self.layer = nn.Sequential(
            nn.Linear(in_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU()
        )
        self.to(cfg.MODEL.DEVICE)
        if ckpt is not None:
            self.load_state_dict(ckpt)

    def forward(self, x):
        return self.layer(x)

class EdgePredictor(nn.Module):
    def __init__(self, cfg, ckpt=None):
        super(EdgePredictor, self).__init__()
        self.pred = nn.Sequential(
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        self.to(cfg.MODEL.DEVICE)
        if ckpt is not None:
            self.load_state_dict(ckpt)

    def forward(self, x_edge):
        return self.pred(x_edge)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = EdgePredictor(cfg=None).to(device)
    summary(model, (1, 6))
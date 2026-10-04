import torch
import torch.nn as nn


class MotionAdapter(nn.Module):
    def __init__(self, dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Start with almost no effect on LTX
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, video_tokens, condition_tokens):
        condition = self.net(condition_tokens)
        return video_tokens + self.scale * condition
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapter import MotionAdapter


B = 1
S = 2025
D = 128

video_tokens = torch.randn(B, S, D)
condition_tokens = torch.randn(B, S, D)

adapter = MotionAdapter(D)

output = adapter(video_tokens, condition_tokens)

print("Video tokens:      ", video_tokens.shape)
print("Condition tokens:  ", condition_tokens.shape)
print("Output:             ", output.shape)
print("Adapter scale:      ", adapter.scale.item())
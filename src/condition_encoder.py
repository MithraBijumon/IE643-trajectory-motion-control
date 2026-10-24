import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """
    Encodes trajectory conditioning maps:

        (B, 3, 33, 480, 864)
                    ↓
        (B, 128, 5, 15, 27)

    Channels:
        0 = heatmap
        1 = dx
        2 = dy
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(3, 32, 3, stride=(2, 2, 2), padding=1),
            nn.SiLU(),

            nn.Conv3d(32, 64, 3, stride=(2, 2, 2), padding=1),
            nn.SiLU(),

            nn.Conv3d(64, 128, 3, stride=(2, 2, 2), padding=1),
            nn.SiLU(),

            nn.Conv3d(128, 128, 3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),

            nn.Conv3d(128, 128, 3, stride=(1, 2, 2), padding=1),
        )

    def forward(self, condition):
        return self.encoder(condition)

class EntityConditionEncoder(nn.Module):
    """
    Encodes the entity-aware condition built on the latent grid:

        (B, 3 + C, F, H', W')  ->  (B, 128, F', H', W')

    Only the time axis is downsampled (F -> (F-1)/8 + 1 through three stride-2 convolutions);
    space is already at the latent resolution.
    """

    def __init__(self, in_channels=3 + 128, hidden=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden, 3, stride=(2, 1, 1), padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, 3, stride=(2, 1, 1), padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, 3, stride=(2, 1, 1), padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, 3, padding=1),
        )

    def forward(self, condition):
        return self.encoder(condition)

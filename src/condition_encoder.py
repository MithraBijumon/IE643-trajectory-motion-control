import torch.nn as nn


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

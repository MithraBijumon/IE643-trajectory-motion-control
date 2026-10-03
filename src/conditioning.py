import torch


def rasterize_trajectories(
    trajectories,
    visible,
    height,
    width,
    sigma=0.03,
    flow_scale=20.0,
):
    """
    trajectories: (B, F, 2) or (B, N, F, 2)
    visible:      (B, F) or (B, N, F)

    Returns:
        (B, F, 3, H, W)

    Channels:
        0 -> point location (heatmap)
        1 -> x movement
        2 -> y movement
    """

    # Add point dimension for single-point input
    if trajectories.ndim == 3:
        trajectories = trajectories.unsqueeze(1)
        visible = visible.unsqueeze(1)

    trajectories = trajectories.float()
    visible = visible.bool()
    device = trajectories.device

    # Movement from frame t -> t+1
    displacement = torch.zeros_like(trajectories)
    displacement[:, :, :-1] = (
        trajectories[:, :, 1:] - trajectories[:, :, :-1]
    )

    valid = torch.zeros_like(visible)
    valid[:, :, :-1] = visible[:, :, 1:] & visible[:, :, :-1]

    displacement *= valid.unsqueeze(-1)
    displacement *= flow_scale

    # Normalized pixel coordinates
    y = (torch.arange(height, device=device) + 0.5) / height
    x = (torch.arange(width, device=device) + 0.5) / width

    aspect = width / height

    dx = (
        x.view(1, 1, 1, 1, width)
        - trajectories[..., 0, None, None]
    ) * aspect

    dy = (
        y.view(1, 1, 1, height, 1)
        - trajectories[..., 1, None, None]
    )

    # Gaussian around each tracked point
    heat = torch.exp(
        -(dx**2 + dy**2) / (2 * sigma**2)
    )

    heat *= visible[..., None, None]

    # Combine multiple points
    heat_sum = heat.sum(dim=1).clamp(min=1e-6)

    flow_x = (
        heat * displacement[..., 0, None, None]
    ).sum(dim=1) / heat_sum

    flow_y = (
        heat * displacement[..., 1, None, None]
    ).sum(dim=1) / heat_sum

    heat = heat.amax(dim=1)

    return torch.stack(
        [
            heat,
            flow_x * heat,
            flow_y * heat,
        ],
        dim=2,
    )
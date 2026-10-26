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

def entity_footprint(trajectory, visible, radius, grid_height, grid_width, min_sigma=0.75):
    """
    Gaussian footprint of the controlled object on the latent grid.

    trajectory: (B, F, 2) normalized object centre per frame
    visible:    (B, F)
    radius:     (B,) object radius as a fraction of sqrt(H*W)

    Returns (B, F, grid_height, grid_width), zero on frames where the object is hidden.
    """
    traj = trajectory.float()
    vis = visible.bool()
    device = traj.device
    B = traj.shape[0]

    cx = traj[..., 0] * grid_width  # in latent cells (square cells, so distances are isotropic)
    cy = traj[..., 1] * grid_height

    radius_cells = radius.float() * (grid_height * grid_width) ** 0.5
    sigma = (0.5 * radius_cells).clamp(min=min_sigma).view(B, 1, 1, 1)

    xs = torch.arange(grid_width, device=device).float() + 0.5
    ys = torch.arange(grid_height, device=device).float() + 0.5
    d2 = (xs.view(1, 1, 1, -1) - cx[..., None, None]) ** 2 + (
        ys.view(1, 1, -1, 1) - cy[..., None, None]
    ) ** 2
    return torch.exp(-d2 / (2 * sigma**2)) * vis[..., None, None]


def render_entity_condition(
    trajectory,
    visible,
    embedding,
    radius,
    grid_height,
    grid_width,
    flow_scale=20.0,
    min_sigma=0.75,
):
    """
    Entity-aware condition rendered directly on the latent grid.

    trajectory: (B, F, 2) normalized object centre per frame
    visible:    (B, F)
    embedding:  (B, C) feature of the object (masked mean of the image latent)
    radius:     (B,) object radius as a fraction of sqrt(H*W)

    The footprint is a Gaussian whose width follows the object size (DragAnything), so the
    object occupies several latent cells instead of a sub-cell dot. The displacement to the
    next frame is painted inside it (Tora-style), and the entity embedding is painted there
    too so the network knows *what* is moving.

    Returns (B, 3 + C, F, grid_height, grid_width): footprint, dx, dy, embedding * footprint.
    """
    traj = trajectory.float()
    vis = visible.bool()
    B = traj.shape[0]

    footprint = entity_footprint(traj, vis, radius, grid_height, grid_width, min_sigma)  # (B, F, H, W)

    disp = torch.zeros_like(traj)
    disp[:, :-1] = traj[:, 1:] - traj[:, :-1]
    valid = torch.zeros_like(vis)
    valid[:, :-1] = vis[:, 1:] & vis[:, :-1]
    disp = disp * valid.unsqueeze(-1) * flow_scale

    dx = footprint * disp[..., 0, None, None]
    dy = footprint * disp[..., 1, None, None]
    entity = footprint[:, None] * embedding.float().view(B, -1, 1, 1, 1)  # (B, C, F, H, W)

    return torch.cat([footprint[:, None], dx[:, None], dy[:, None], entity], dim=1)

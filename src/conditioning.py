import torch
import torch.nn.functional as F


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


SHAPE_FACTOR = 4   # the mask is kept at 4x the latent grid, so sub-cell objects keep their shape


def shape_footprint(mask, trajectory, visible, grid_height, grid_width, gauss=None, gauss_weight=0.3):
    """Footprint with the object's real silhouette.

    mask:       (B, 1, SHAPE_FACTOR*grid_height, SHAPE_FACTOR*grid_width) coverage (0-1) of the object in the
                conditioning frame (the first frame of the clip)
    trajectory: (B, F, 2) normalized; the silhouette is translated by trajectory[t] - trajectory[0]
                (rigid, so it can be built at inference from just the first-frame mask)
    gauss:      optional (B, F, H, W) Gaussian footprint, blended in so a sub-cell object never vanishes

    Returns (B, F, grid_height, grid_width) in [0, 1], zero where the object is hidden.
    """
    B, Fr = trajectory.shape[:2]
    delta = trajectory.float() - trajectory.float()[:, :1]                       # (B, F, 2)
    theta = torch.zeros(B * Fr, 2, 3, device=mask.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1.0
    theta[:, :, 2] = -2.0 * delta.reshape(-1, 2)                                 # output(p) = input(p - delta)
    src = mask.float().repeat_interleave(Fr, dim=0)                              # (B*F, 1, Hh, Wh)
    grid = F.affine_grid(theta, src.shape, align_corners=False)
    moved = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    pooled = F.avg_pool2d(moved, SHAPE_FACTOR).view(B, Fr, grid_height, grid_width)

    peak = pooled.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    fp = pooled / peak                                                           # a small object still reaches 1
    if gauss is not None:
        fp = fp + gauss_weight * gauss
    return fp.clamp(max=1.0) * visible.bool()[..., None, None]


def render_entity_condition(
    trajectory,
    visible,
    embedding,
    radius,
    grid_height,
    grid_width,
    flow_scale=20.0,
    min_sigma=0.75,
    shape=None,
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
    if shape is not None:  # real silhouette (translated along the path) instead of a round blob
        footprint = shape_footprint(shape, traj, vis, grid_height, grid_width, gauss=footprint)

    disp = torch.zeros_like(traj)
    disp[:, :-1] = traj[:, 1:] - traj[:, :-1]
    valid = torch.zeros_like(vis)
    valid[:, :-1] = vis[:, 1:] & vis[:, :-1]
    disp = disp * valid.unsqueeze(-1) * flow_scale

    dx = footprint * disp[..., 0, None, None]
    dy = footprint * disp[..., 1, None, None]
    entity = footprint[:, None] * embedding.float().view(B, -1, 1, 1, 1)  # (B, C, F, H, W)

    return torch.cat([footprint[:, None], dx[:, None], dy[:, None], entity], dim=1)

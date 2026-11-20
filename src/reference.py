import cv2
import numpy as np


def render_reference(image, mask, trajectory, visible=None, feather=1.5):
    """A rough "drag video": the object's cut-out moved rigidly along the trajectory over the scene.

    image:      (H, W, 3) uint8, the conditioning frame
    mask:       (H, W) bool, the object in that frame
    trajectory: (F, 2) normalized object centre per frame (frame 0 = where the mask is)
    visible:    (F,) bool, frames where the object is hidden show only the scene

    The object's hole is filled with OpenCV inpainting. Frame 0 is the input image itself.
    Returns (F, H, W, 3) uint8.
    """
    H, W = image.shape[:2]
    F = len(trajectory)
    mask = np.asarray(mask, bool)
    if visible is None:
        visible = np.ones(F, bool)
    if not mask.any():
        return np.repeat(image[None], F, axis=0)

    hole = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    scene = cv2.inpaint(image, hole, 5, cv2.INPAINT_TELEA)
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather)[..., None]

    traj = np.asarray(trajectory, np.float32)
    out = np.empty((F, H, W, 3), np.uint8)
    for f in range(F):
        if not visible[f]:
            out[f] = scene
            continue
        dx, dy = (traj[f] - traj[0]) * (W, H)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        rgb = cv2.warpAffine(image, m, (W, H), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32)
        a = cv2.warpAffine(alpha, m, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
        out[f] = (scene * (1 - a) + rgb * a).round().astype(np.uint8)
    out[0] = image
    return out

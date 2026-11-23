## overall
| method | ate | epe_final | pck_5 | pck_10 | direction_cos | progress | clip_image_sim | first_frame_psnr | first_frame_ssim | color_drift | clip_temporal | warp_error | sharpness_ratio | aesthetic | obj_ate | obj_direction_cos | obj_progress | obj_clip_sim | obj_area_ratio | bg_drift | bg_flow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.157 | 0.297 | 0.170 | 0.337 | 0.054 | 0.008 | 0.943 | 33.511 | 0.936 | 0.013 | 0.995 | 0.008 | 0.730 | nan | 0.155 | 0.096 | 0.025 | 0.902 | 1.087 | 0.062 | 0.003 |
| adapter | 0.137 | 0.272 | 0.213 | 0.400 | 0.268 | 0.108 | 0.943 | 33.530 | 0.937 | 0.014 | 0.994 | 0.008 | 0.700 | nan | 0.140 | 0.195 | 0.097 | 0.906 | 1.016 | 0.059 | 0.003 |

## per motion (opposite directions show forward / backward asymmetry)
| method | ate | direction_cos | progress | obj_ate | obj_direction_cos | bg_drift | bg_flow |
|---|---|---|---|---|---|---|---|
| baseline / left | 0.137 | 0.068 | 0.104 | 0.151 | -0.159 | 0.062 | 0.003 |
| adapter / left | 0.138 | 0.073 | 0.050 | 0.150 | -0.161 | 0.061 | 0.003 |
| baseline / right | 0.155 | -0.068 | -0.104 | 0.163 | 0.159 | 0.062 | 0.003 |
| adapter / right | 0.125 | 0.371 | 0.189 | 0.138 | 0.351 | 0.059 | 0.002 |
| baseline / up | 0.194 | -0.450 | -0.162 | 0.186 | -0.518 | 0.062 | 0.003 |
| adapter / up | 0.156 | -0.066 | -0.033 | 0.165 | -0.317 | 0.059 | 0.003 |
| baseline / down | 0.147 | 0.450 | 0.162 | 0.133 | 0.518 | 0.062 | 0.003 |
| adapter / down | 0.140 | 0.364 | 0.127 | 0.128 | 0.513 | 0.058 | 0.002 |
| baseline / diag | 0.152 | 0.270 | 0.041 | 0.143 | 0.479 | 0.062 | 0.003 |
| adapter / diag | 0.124 | 0.597 | 0.204 | 0.120 | 0.589 | 0.059 | 0.002 |

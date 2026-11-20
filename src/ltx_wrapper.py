import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.conditioning import SHAPE_FACTOR, entity_footprint
from src.motion_control import MotionControl, pack_latents, unpack_latents
from src.reference import render_reference
from src.segmentation import disk_mask
from src.utils import torch_dtype, video_tensor_to_uint8

VAE_SPATIAL = 32
VAE_TEMPORAL = 8


def linear_quadratic_sigmas(num_steps, threshold_noise=0.025):
    """Sigma schedule LTX-Video was tuned for (1.0 -> ~0), with a terminal 0 appended."""
    from diffusers.pipelines.ltx.pipeline_ltx_condition import linear_quadratic_schedule

    sigmas = linear_quadratic_schedule(num_steps, threshold_noise).float()
    return torch.cat([sigmas, torch.zeros(1)])


class LTXBackbone:
    """Frozen LTX-Video (VAE + transformer + cached text embeddings) together with
    the trainable MotionControl module.

    Training follows LTX's flow-matching formulation:
        x_sigma = (1 - sigma) * x0 + sigma * noise
        the transformer predicts the velocity (noise - x0)
    (LTXConditionPipeline steps x += (sigma_next - sigma) * pred with per-token timesteps)
    The first latent frame is the clean conditioning image (timestep 0, no loss),
    exactly like LTXConditionPipeline's image-to-video mode.
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = torch_dtype(cfg["model"]["dtype"])

        if cfg["is_tiny"]:
            self._build_tiny()
        else:
            self._build_pretrained()

        self.vae.requires_grad_(False).eval()
        self.transformer.requires_grad_(False).eval()

        if cfg["model"].get("gradient_checkpointing") and not cfg["is_tiny"]:
            self.transformer.enable_gradient_checkpointing()
        if cfg["model"].get("vae_tiling"):
            self.vae.enable_tiling()

        a = cfg.get("adapter", {})
        tc = self.transformer.config
        self.motion = MotionControl(
            dim=tc.in_channels,
            num_blocks=len(self.transformer.transformer_blocks),
            hidden_dim=tc.num_attention_heads * tc.attention_head_dim,
            inject_every=a.get("inject_every", 4),
            relative=a.get("relative", False),
            max_inject=a.get("max_inject", 0.0),
            cond_blocks=a.get("cond_blocks", 0),
            cond_heads=a.get("cond_heads", 4),
            local_inject=a.get("local_inject", 0.0),
            reference=a.get("reference", False),
        ).to(self.device)
        self.motion.attach(self.transformer)

        self.register_latent_stats()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_pretrained(self):
        from diffusers import LTXConditionPipeline

        m = self.cfg["model"]
        pipe = LTXConditionPipeline.from_pretrained(m["id"], torch_dtype=self.dtype)

        # The prompt is fixed, so encode it once and drop the 4.7B T5 encoder.
        pipe.text_encoder.to(self.device)
        with torch.no_grad():
            pos, pos_mask, neg, neg_mask = pipe.encode_prompt(
                m["prompt"],
                m["negative_prompt"],
                do_classifier_free_guidance=True,
                device=self.device,
            )
        self.prompt = (pos, pos_mask)
        self.negative_prompt = (neg, neg_mask)

        pipe.text_encoder.to("cpu")
        del pipe.text_encoder
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self.vae = pipe.vae.to(self.device)
        self.transformer = pipe.transformer.to(self.device)

    def _build_tiny(self):
        from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel

        g = torch.Generator().manual_seed(0)
        torch.manual_seed(0)

        self.vae = AutoencoderKLLTXVideo(
            block_out_channels=(8, 8, 8, 8),
            decoder_block_out_channels=(8, 8, 8, 8),
            layers_per_block=(1, 1, 1, 1, 1),
            decoder_layers_per_block=(1, 1, 1, 1, 1),
        ).to(self.device, self.dtype)

        self.transformer = LTXVideoTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            cross_attention_dim=32,
            num_layers=2,
            caption_channels=32,
        ).to(self.device, self.dtype)

        emb = torch.randn(1, 16, 32, generator=g)
        mask = torch.ones(1, 16, dtype=torch.long)
        self.prompt = (emb.to(self.device, self.dtype), mask.to(self.device))
        self.negative_prompt = (torch.zeros_like(self.prompt[0]), mask.to(self.device))

    def register_latent_stats(self):
        self.latents_mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).float().to(self.device)
        self.latents_std = self.vae.latents_std.view(1, -1, 1, 1, 1).float().to(self.device)
        self.scaling = float(self.vae.config.scaling_factor)

    # ------------------------------------------------------------------
    # Latent helpers
    # ------------------------------------------------------------------

    def normalize(self, z):
        return (z - self.latents_mean) * self.scaling / self.latents_std

    def denormalize(self, z):
        return z * self.latents_std / self.scaling + self.latents_mean

    @torch.no_grad()
    def encode_video(self, video):
        """(B, 3, F, H, W) in [-1, 1] -> normalized fp32 latents (B, 128, F', H', W')."""
        z = self.vae.encode(video.to(self.device, self.dtype)).latent_dist.sample()
        return self.normalize(z.float())

    @torch.no_grad()
    def decode_latents(self, latents, decode_timestep=0.05, generator=None):
        """Normalized latents (B, 128, F', H', W') -> uint8 frames (B, F, H, W, 3)."""
        z = self.denormalize(latents.float()).to(self.dtype)
        timestep = None

        if self.vae.config.timestep_conditioning:
            noise = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
            z = (1 - decode_timestep) * z + decode_timestep * noise
            timestep = torch.full((z.shape[0],), decode_timestep, device=z.device, dtype=z.dtype)

        video = self.vae.decode(z, timestep, return_dict=False)[0]
        return video_tensor_to_uint8(video)

    def entity_features(self, first_latent, mask, area, training=False):
        """The controlled object as the network sees it.

        first_latent: (B, 128, H', W') normalized latent of the conditioning frame
        mask:         (B, 1, H, W) float, the object in that frame
        area:         (B,) object area as a fraction of the frame

        embedding = mean of the image latent inside the mask; radius is that of a disk with
        the same area, as a fraction of sqrt(H*W).
        """
        w = F.adaptive_avg_pool2d(mask.float(), first_latent.shape[-2:])
        embedding = (w * first_latent).sum((2, 3)) / w.sum((2, 3)).clamp(min=1e-6)
        a = self.cfg.get("adapter", {})
        if not a.get("entity_embedding", True):
            embedding = torch.zeros_like(embedding)
        elif training and a.get("embedding_dropout", 0.0) > 0:
            # an appearance vector lets the adapter recognise (memorise) scenes; hiding it for some
            # samples forces it to also work from the footprint and the motion alone
            keep = (torch.rand(embedding.shape[0], 1, device=embedding.device) >= a["embedding_dropout"]).float()
            embedding = embedding * keep

        out = {"embedding": embedding, "radius": (area.float() / math.pi).sqrt()}
        if a.get("mask_shape", True):
            Hl, Wl = first_latent.shape[-2:]
            out["shape"] = F.adaptive_avg_pool2d(mask.float(), (SHAPE_FACTOR * Hl, SHAPE_FACTOR * Wl))
        return out

    def rope_fps(self):
        """Frame rate used for the rotary time coordinates (the data's real fps if you set data.rope_fps)."""
        return self.cfg["data"].get("rope_fps", 24)

    def video_coords(self, batch, latent_shape, frame_rate=None):
        """Rotary position coordinates, same construction as LTXConditionPipeline."""
        frame_rate = frame_rate or self.rope_fps()
        Fl, Hl, Wl = latent_shape
        coords = torch.meshgrid(
            torch.arange(Fl), torch.arange(Hl), torch.arange(Wl), indexing="ij"
        )
        coords = torch.stack(coords, 0).reshape(1, 3, -1).repeat(batch, 1, 1).float()

        coords = coords * torch.tensor([VAE_TEMPORAL, VAE_SPATIAL, VAE_SPATIAL]).view(1, 3, 1)
        coords[:, 0] = (coords[:, 0] + 1 - VAE_TEMPORAL).clamp(min=0)
        coords[:, 0] = coords[:, 0] / frame_rate
        return coords.to(self.device)

    @staticmethod
    def first_frame_mask(batch, latent_shape, device):
        """(B, S) bool: True for tokens of the first latent frame (the clean image)."""
        Fl, Hl, Wl = latent_shape
        mask = torch.zeros(batch, Fl, Hl * Wl, dtype=torch.bool, device=device)
        mask[:, 0] = True
        return mask.reshape(batch, -1)

    def _text(self, batch, negative=False):
        emb, mask = self.negative_prompt if negative else self.prompt
        return emb.expand(batch, -1, -1), mask.expand(batch, -1)

    def _predict(self, tokens, sigma_tokens, keep, coords, negative=False):
        """One transformer call. sigma_tokens: (B, S) in [0, 1]."""
        B = tokens.shape[0]
        emb, mask = self._text(B, negative)

        # conditioned (image) tokens always see timestep 0
        timestep = (sigma_tokens * 1000.0).masked_fill(keep, 0.0)

        out = self.transformer(
            hidden_states=tokens.to(self.dtype),
            encoder_hidden_states=emb,
            timestep=timestep.float(),
            encoder_attention_mask=mask,
            video_coords=coords,
            return_dict=False,
        )[0]
        return out.float()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_loss(self, batch, generator=None, use_condition=None, sigma=None, focus=None, return_parts=False):
        """use_condition: None = normal (with condition dropout), True/False = force the
        trajectory on/off for every sample (used to measure what the trajectory is worth).
        focus: extra loss weight on the tokens around the controlled object, w = 1 + focus * footprint
        (None = train.focus_loss from the config, 0 = plain mean). Validation passes 0 so that
        cond_gain stays comparable between runs.
        return_parts: also return {"obj": ..., "bg": ...}, the plain mean error inside / outside the object mask."""
        video = batch["video"].permute(0, 2, 1, 3, 4)  # B,F,3,H,W -> B,3,F,H,W
        trajectory = batch["trajectory"].to(self.device)
        visible = batch["visible"].to(self.device)

        x0 = self.encode_video(video)
        B, _, Fl, Hl, Wl = x0.shape
        latent_shape = (Fl, Hl, Wl)

        x0_tokens = pack_latents(x0)
        noise = torch.randn(x0_tokens.shape, generator=generator, device=self.device)

        # logit-normal sigma sampling, as in LTX / SD3 flow-matching training
        # train.sigma_mean > 0 shifts it towards noisier steps, where the motion is decided
        mean = self.cfg["train"].get("sigma_mean", 0.0)
        drawn = torch.sigmoid(torch.randn(B, 1, generator=generator, device=self.device) + mean)
        # `sigma` (a float in (0, 1)) pins the noise level, e.g. for a validation grid
        sigma = drawn if sigma is None else torch.full_like(drawn, float(sigma))
        keep = self.first_frame_mask(B, latent_shape, self.device)

        noisy = (1 - sigma.unsqueeze(-1)) * x0_tokens + sigma.unsqueeze(-1) * noise
        noisy = torch.where(keep.unsqueeze(-1), x0_tokens, noisy)

        # condition dropout: some samples train without the trajectory so that
        # generation can later amplify its effect (motion guidance)
        p_drop = self.cfg["train"].get("cond_dropout", 0.0)
        cond_mask = (torch.rand(B, generator=generator, device=self.device) >= p_drop).float()
        if use_condition is not None:
            cond_mask = torch.full((B,), float(use_condition), device=self.device)

        entity = self.entity_features(
            x0[:, :, 0], batch["mask"].to(self.device), batch["mask_area"].to(self.device),
            training=use_condition is None,  # embedding dropout only in real training steps, not in validation
        )

        if self.motion.reference:
            entity["ref_latent"] = self.encode_video(batch["reference"].to(self.device).permute(0, 2, 1, 3, 4).float() / 127.5 - 1)

        res = self.motion.residuals(
            trajectory, visible, latent_shape, keep_mask=keep, cond_mask=cond_mask, entity=entity
        )
        # the hooks read this during forward AND during the checkpointed
        # recompute in backward, so it is deliberately left set after this returns
        self.motion.set_context(res)

        pred = self._predict(
            noisy,
            sigma.expand(-1, noisy.shape[1]),
            keep,
            self.video_coords(B, latent_shape),
        )

        target = noise - x0_tokens
        err = ((pred - target) ** 2).mean(dim=-1)  # (B, S)

        t = self.cfg["train"]
        focus = t.get("focus_loss", 0.0) if focus is None else focus

        # where the controlled object is, per latent token, in [0, 1]
        if "loss_masks" in batch:
            # the object's TRUE per-frame mask (SAM 2), widened by one cell, on the latent grid.
            # Training only: nothing here has to exist at inference.
            lm = batch["loss_masks"].to(self.device).float()               # (B, F, h, w)
            idx = [min(VAE_TEMPORAL * k, lm.shape[1] - 1) for k in range(Fl)]
            near = F.adaptive_max_pool2d(lm[:, idx].flatten(0, 1), (Hl, Wl))
            obj = F.max_pool2d(near, 3, stride=1, padding=1).view(B, Fl, Hl, Wl).reshape(B, -1)
        elif focus > 0 or return_parts:
            radius = entity["radius"] if entity is not None else torch.full((B,), 0.06, device=self.device)
            fp = entity_footprint(trajectory, visible, radius, Hl, Wl)  # (B, F, Hl, Wl)
            # latent frame k shows video frame 8k (frame 0 is the clean image frame)
            idx = [min(VAE_TEMPORAL * k, fp.shape[1] - 1) for k in range(Fl)]
            obj = fp[:, idx].reshape(B, -1)
        else:
            obj = None

        valid = (~keep).float()
        if focus > 0 and obj is not None:
            # later frames matter most for following the path (the object is furthest from the
            # conditioning frame there): time_ramp 1 = last frame's object tokens weigh twice the first's
            ramp = t.get("time_ramp", 0.0)
            tw = 1.0
            if ramp > 0:
                tw = 1 + ramp * torch.linspace(0, 1, Fl, device=self.device).view(1, Fl, 1).expand(1, Fl, Hl * Wl).reshape(1, -1)
            if t.get("split_loss", False):
                # object and background are averaged separately, so the object term cannot be diluted by the
                # (far larger) background, and the background term is an explicit "keep the scene" signal
                w_obj = valid * obj * tw
                w_bg = valid * (1 - obj)
                l_obj = (err * w_obj).sum() / w_obj.sum().clamp(min=1e-6)
                l_bg = (err * w_bg).sum() / w_bg.sum().clamp(min=1e-6)
                loss = (focus * l_obj + l_bg) / (focus + 1)
            else:
                w = valid * (1 + focus * obj * tw)
                loss = (err * w).sum() / w.sum()
        else:
            loss = (err * valid).sum() / valid.sum()

        if return_parts:
            # unweighted mean error inside / outside the object, whatever the training weighting is
            with torch.no_grad():
                inside, outside = valid * (obj > 0.5).float(), valid * (obj <= 0.5).float()
                parts = {
                    "obj": float((err * inside).sum() / inside.sum().clamp(min=1.0)),
                    "bg": float((err * outside).sum() / outside.sum().clamp(min=1.0)),
                }
            return loss, parts
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        image,
        trajectory,
        visible=None,
        steps=30,
        guidance_scale=1.0,
        decode_timestep=0.05,
        seed=0,
        progress=None,
        motion_guidance=1.0,
        use_motion=True,
        mask=None,
        ref_init=None,
    ):
        """
        image:      PIL image, already sized (width, height) = cfg data size
        trajectory: (F, 2) normalized, F = cfg num_frames
        motion_guidance: >1 amplifies the trajectory's effect (needs cond_dropout in training)
        use_motion: False runs plain LTX image-to-video (the baseline)
        mask: (H, W) bool mask of the controlled object in `image`;
              a disk around the first trajectory point is used when it is missing
        ref_init: start the denoising from the drag video (src/reference.py) noised to this sigma (0 = from pure noise;
                  None = inference.ref_init). Only with use_motion; the baseline never sees it.
        Returns uint8 frames (F, H, W, 3).
        """
        d = self.cfg["data"]
        H, W, nF = d["height"], d["width"], d["num_frames"]

        if image.size != (W, H):
            image = image.resize((W, H), Image.BICUBIC)
        if len(trajectory) != nF:
            raise ValueError(f"trajectory must have {nF} frames, got {len(trajectory)}")
        if visible is None:
            visible = np.ones(nF, dtype=bool)

        gen = torch.Generator(device=self.device).manual_seed(seed)

        img = torch.from_numpy(np.array(image)).float().div(127.5).sub(1)
        img = img.permute(2, 0, 1)[None, :, None]  # 1,3,1,H,W
        img_latent = pack_latents(self.encode_video(img))  # 1, Hl*Wl, 128

        Fl, Hl, Wl = (nF - 1) // VAE_TEMPORAL + 1, H // VAE_SPATIAL, W // VAE_SPATIAL
        latent_shape = (Fl, Hl, Wl)
        S = Fl * Hl * Wl

        keep = self.first_frame_mask(1, latent_shape, self.device)
        coords = self.video_coords(1, latent_shape)

        traj = torch.as_tensor(np.asarray(trajectory), dtype=torch.float32, device=self.device)[None]
        vis = torch.as_tensor(np.asarray(visible), dtype=torch.bool, device=self.device)[None]
        if mask is None:
            mask = disk_mask(H, W, np.asarray(trajectory)[0], 0.06)
        elif mask.shape != (H, W):
            mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
        mask_t = torch.from_numpy(mask).float().to(self.device)[None, None]
        image_latent = unpack_latents(img_latent, 1, Hl, Wl)[:, :, 0]
        entity = self.entity_features(image_latent, mask_t, mask_t.mean().view(1))

        if ref_init is None:
            ref_init = self.cfg["inference"].get("ref_init", 0.0)
        ref_tokens = None
        if use_motion and (self.motion.reference or ref_init > 0):
            drag = render_reference(np.array(image), mask, np.asarray(trajectory), np.asarray(visible))
            drag = torch.from_numpy(drag).float().div(127.5).sub(1).permute(3, 0, 1, 2)[None]  # 1,3,F,H,W
            ref_latent = self.encode_video(drag)
            entity["ref_latent"] = ref_latent
            ref_tokens = pack_latents(ref_latent)

        res = self.motion.residuals(traj, vis, latent_shape, keep_mask=keep, entity=entity)

        def velocity(x, sig, motion_on):
            self.motion.set_context(res if motion_on else None)
            pred = self._predict(x, sig, keep, coords)
            if guidance_scale > 1.0:
                uncond = self._predict(x, sig, keep, coords, negative=True)
                pred = uncond + guidance_scale * (pred - uncond)
            return pred

        x = torch.randn(1, S, img_latent.shape[-1], generator=gen, device=self.device)
        x[:, : img_latent.shape[1]] = img_latent

        sigmas = linear_quadratic_sigmas(steps).to(self.device)

        first = 0
        if ref_tokens is not None and ref_init > 0:
            # SDEdit-style start: the drag video, noised to sigma = ref_init, instead of pure noise
            first = int((sigmas[:-1] > ref_init).sum())
            s0 = float(sigmas[first])
            x = (1 - s0) * ref_tokens + s0 * torch.randn(ref_tokens.shape, generator=gen, device=self.device)
            x[:, : img_latent.shape[1]] = img_latent

        for i in range(first, steps):
            s, s_next = sigmas[i], sigmas[i + 1]
            sig = torch.full((1, S), float(s), device=self.device)

            pred = velocity(x, sig, use_motion)
            if use_motion and motion_guidance != 1.0:
                off = velocity(x, sig, False)
                pred = off + motion_guidance * (pred - off)

            # pred is the velocity (noise - x0); sigma decreases, so s_next - s < 0
            x_new = x + (s_next - s) * pred
            x = torch.where(keep.unsqueeze(-1), x, x_new)

            if progress is not None:
                progress((i + 1) / steps)

        self.motion.set_context(None)
        latents = unpack_latents(x, Fl, Hl, Wl)
        return self.decode_latents(latents, decode_timestep, generator=gen)[0]

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, path, **extra):
        torch.save({**self.motion.trainable_state(), "cfg": self.cfg, **extra}, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.motion.load_trainable_state(ckpt)
        return ckpt

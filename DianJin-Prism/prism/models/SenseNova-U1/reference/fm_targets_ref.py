"""Read-only oracle: the official flow-matching TARGET math (FM), ported verbatim.

This is the parity oracle for the gen-side FM construction (the achievable part of
plan §8's "L3"; a FULL numeric parity vs the InternEvo trainer would need a live
official single-process harness, which is out of scope here). NOT imported by the
runtime — only the parity tests import it.

Provenance (prism/models/SenseNova-U1/reference/):
  - modeling_sensenovavl_chat_mot.py::_apply_time_schedule  :562-580 (standard branch;
    HF port modeling_neo_chat.py:427-447 uses the passed timestep_shift)
  - modeling_sensenovavl_chat_mot.py::prepare_image_gen_targets :582-673
        z = t·x + (1-t)·(randn·noise_scale) ; t = time_schedule(sigmoid(u))
        noise_scale embedding value = noise_scale / noise_scale_max_value
  - modeling_sensenovavl_chat_mot.py::image_gen_loss :1399-1444
        v = (pred_x - z) / (1 - t).clamp_min(t_eps) ; per-image weight = 1/sqrt(seq_len)
  - flow_grpo/neounify/inferencer.py::_compute_noise_scale is the SELF-WRITTEN
    INFERENCE helper (caps at max_value) — NOT authoritative; the FM training-target
    noise_scale is the UNCAPPED prepare_image_gen_targets:619-623 formula above.
"""

from __future__ import annotations

import math

import torch


def apply_time_schedule_standard(t: torch.Tensor, timestep_shift: float) -> torch.Tensor:
    """Standard (Flux-style) shift on sigma=1-t. Verbatim from the standard branch."""
    sigma = 1 - t
    shift = timestep_shift
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return 1 - sigma


def resolution_noise_scale(
    grid_h: int, grid_w: int, merge_size: int, base_image_seq_len: float, noise_scale: float,
) -> float:
    """FM-target z-noise scale, per official TRAINING prepare_image_gen_targets:619-623:
    merged_seq_len = (grid_h*grid_w)//merge**2 ; sqrt(merged/base)*noise_scale.

    NOT capped: official training leaves the z-noise uncapped (only the noise_scale
    EMBEDDING value uses /noise_scale_max_value, separately). The self-written inference
    inferencer caps at max_value for sampling stability — that is NOT the training target."""
    merged = (grid_h * grid_w) // (merge_size ** 2)
    return math.sqrt(merged / float(base_image_seq_len)) * float(noise_scale)


def fm_z(x: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Noised latent z_t = t·x + (1-t)·eps  (eps already scaled by noise_scale)."""
    return t * x + (1 - t) * eps


def fm_v(x: torch.Tensor, z: torch.Tensor, t: torch.Tensor, t_eps: float = 0.02) -> torch.Tensor:
    """Flow-matching velocity target v = (x - z)/(1-t), clamped denominator."""
    return (x - z) / (1 - t).clamp_min(t_eps)


def image_gen_weight(seq_len: int) -> float:
    """Per-image FM loss weight = 1/sqrt(merged seq_len)."""
    return 1.0 / (seq_len ** 0.5)

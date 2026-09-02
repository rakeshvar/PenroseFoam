"""Radial-rank masked flow."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import torch

PROJECT_DIR = Path(__file__).resolve().parent
SPUR_DIR = Path(
    os.environ.get("PENROSE_SPUR_PATH", PROJECT_DIR.parent / "PenroseSpur")
).resolve()
if str(SPUR_DIR) not in sys.path:
    sys.path.append(str(SPUR_DIR))
from flow_geometry import ANGLE_HALF_PERIOD, angle_delta, wrap_angle  # noqa: E402

FLOW_VERSION = "anchor-zero-angle-quantized-polar-rank-v3"


def representative_neighbor_factor(symmetry: int) -> float:
    if symmetry == 6:
        return math.sqrt(3.0)
    if symmetry == 5:
        return (
            0.24 * math.sin(math.pi / 5.0)
            + 0.59 * math.sin(3.0 * math.pi / 10.0)
            + 0.17 * math.sin(2.0 * math.pi / 5.0)
        )
    raise ValueError(f"symmetry must be 5 or 6, got {symmetry}")


def canonicalize_anchor_frame(
    data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Put the nearest-to-center tile in slot zero and express XYA relative to it."""
    if data.ndim != 3 or data.shape[-1] != 3 or data.shape[1] < 1:
        raise ValueError("data must have shape (B,N,3) with N >= 1")
    batch_size, num_tiles = data.shape[:2]
    anchor_indices = data[..., :2].square().sum(dim=-1).argmin(dim=1)
    order = torch.arange(num_tiles, device=data.device).expand(batch_size, -1).clone()
    rows = torch.arange(batch_size, device=data.device)
    order[rows, 0] = anchor_indices
    order[rows, anchor_indices] = 0
    ordered = data.gather(1, order[..., None].expand_as(data))

    anchor_xy = ordered[:, :1, :2]
    anchor_angle = ordered[:, 0, 2]
    relative_xy = ordered[..., :2] - anchor_xy
    phase = anchor_angle * (math.pi / ANGLE_HALF_PERIOD)
    cosine, sine = phase.cos()[:, None], phase.sin()[:, None]
    canonical_xy = torch.stack(
        (
            cosine * relative_xy[..., 0] + sine * relative_xy[..., 1],
            -sine * relative_xy[..., 0] + cosine * relative_xy[..., 1],
        ),
        dim=-1,
    )
    canonical_angle = angle_delta(ordered[..., 2], anchor_angle[:, None])
    canonical = torch.cat((canonical_xy, canonical_angle[..., None]), dim=-1)
    canonical[:, 0] = 0.0
    return canonical, order, anchor_angle


class MaskedFlow:
    """Linear-time XYA path gated by exact-endpoint logistic occupancy."""

    def __init__(self, symmetry: int, side: float, kappa: float = 0.05):
        if side <= 0 or kappa <= 0:
            raise ValueError("side and kappa must be positive")
        self.symmetry = int(symmetry)
        self.side = float(side)
        self.kappa = float(kappa)
        self.radial_bin_width = (
            representative_neighbor_factor(self.symmetry) * self.side / 2.0
        )

    def rank_coordinate(self, data: torch.Tensor) -> torch.Tensor:
        x, y = data[..., 0], data[..., 1]
        radius = torch.sqrt(x.square() + y.square())
        radial_shell = torch.round(radius / self.radial_bin_width)
        phi = torch.remainder(torch.atan2(y, x), 2.0 * math.pi)

        phi_order = torch.argsort(phi, dim=-1, stable=True)
        shells_in_phi_order = radial_shell.gather(1, phi_order)
        shell_order = torch.argsort(shells_in_phi_order, dim=-1, stable=True)
        order = phi_order.gather(1, shell_order)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(
            data.shape[1], device=data.device, dtype=order.dtype
        ).expand_as(order)
        ranks.scatter_(1, order, rank_values)
        return (ranks.to(data.dtype) + 0.5) / data.shape[1]

    def occupancy(
        self, time: torch.Tensor, rho: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if time.ndim != 1 or time.shape[0] != rho.shape[0]:
            raise ValueError("time must have shape (B,)")
        t = time[:, None].to(device=rho.device, dtype=rho.dtype)
        start = torch.sigmoid(-rho / self.kappa)
        end = torch.sigmoid((1.0 - rho) / self.kappa)
        current = torch.sigmoid((t - rho) / self.kappa)
        denominator = (end - start).clamp_min(torch.finfo(rho.dtype).tiny)
        u = (current - start) / denominator
        du = current * (1.0 - current) / (self.kappa * denominator)
        u = torch.where(t == 0, torch.zeros_like(u), u)
        u = torch.where(t == 1, torch.ones_like(u), u)
        return u.clamp(0.0, 1.0), du.clamp_min(0.0)

    def interpolate(
        self, data: torch.Tensor, noise: torch.Tensor, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if data.shape != noise.shape or data.ndim != 3 or data.shape[-1] != 3:
            raise ValueError("data and noise must share shape (B,N,3)")
        rho = torch.zeros_like(data[..., 0])
        if data.shape[1] > 1:
            rho[:, 1:] = self.rank_coordinate(data[:, 1:])
        u, du = self.occupancy(time, rho)
        u = u.clone()
        du = du.clone()
        u[:, 0] = 1.0
        du[:, 0] = 0.0
        xy_delta = data[..., :2] - noise[..., :2]
        a_delta = angle_delta(data[..., 2], noise[..., 2])
        state_xya = torch.cat(
            (
                noise[..., :2] + u[..., None] * xy_delta,
                wrap_angle(noise[..., 2] + u * a_delta)[..., None],
            ),
            dim=-1,
        )
        delta = torch.cat((xy_delta, a_delta[..., None]), dim=-1)
        velocity = torch.cat((du[..., None] * delta, du[..., None]), dim=-1)
        return torch.cat((state_xya, u[..., None]), dim=-1), velocity

    @staticmethod
    def sample_training_times(
        batch_size: int, device: torch.device, generator: torch.Generator
    ) -> torch.Tensor:
        return torch.rand(batch_size, device=device, generator=generator)

    @staticmethod
    def sampling_times(
        num_steps: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        return torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)

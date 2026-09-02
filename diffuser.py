"""Radial-rank masked flow."""

from __future__ import annotations

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
from flow_geometry import angle_delta, wrap_angle  # noqa: E402


class MaskedFlow:
    """Linear-time XYA path gated by exact-endpoint logistic occupancy."""

    def __init__(self, kappa: float = 0.05):
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        self.kappa = float(kappa)

    @staticmethod
    def rank_coordinate(data: torch.Tensor) -> torch.Tensor:
        radius_sq = data[..., :2].square().sum(dim=-1)
        order = torch.argsort(radius_sq, dim=-1, stable=True)
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
        rho = self.rank_coordinate(data)
        u, du = self.occupancy(time, rho)
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

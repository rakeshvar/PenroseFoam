"""Train PenroseFoam's single masked optimal-transport flow."""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR

from checkpoint import capture_rng, restore_rng, retain_newest_and_best, save_epoch
from config import (
    Config, batches_per_epoch, load_config, make_identifier, make_run_name,
    validate_resume_config,
)
from denoiser import DirectTransformer
from diffuser import FLOW_VERSION, MaskedFlow
from sampler import (
    build_spur, choose_device, lattice_loss, make_generator, prepare_flow_batch,
    reverse_sample, save_sample_svg,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Config,
    schedule_epochs: int | None = None,
) -> LambdaLR:
    epochs = schedule_epochs or config.train.num_epochs
    warmup = (
        config.train.warmup_epochs
        if config.train.warmup_epochs is not None
        else min(10, math.floor(0.05 * epochs))
    )
    if warmup >= epochs:
        raise ValueError("warmup epochs must be less than scheduled training epochs")
    floor = config.train.min_lr_factor

    def factor(position: int) -> float:
        if warmup and position <= warmup:
            return 0.01 + 0.99 * position / warmup
        if position <= epochs:
            progress = (position - warmup) / max(1, epochs - warmup)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
        return floor

    return LambdaLR(optimizer, factor)


class WandbLogger:
    def __init__(
        self, config: Config, run_name: str, run_id: str | None,
        resume: bool, parameter_counts: dict[str, int],
    ):
        self.run = None
        if not config.wandb.enable or not os.environ.get("WANDB_API_KEY"):
            return
        try:
            import wandb
        except ModuleNotFoundError:
            return
        kwargs: dict[str, Any] = {
            "project": config.wandb.project,
            "name": run_name,
            "config": {**config.to_dict(), "parameter_counts": parameter_counts},
        }
        if run_id:
            kwargs.update(id=run_id, resume="must" if resume else "never")
        self.run = wandb.init(**kwargs)

    @property
    def run_id(self) -> str | None:
        return None if self.run is None else self.run.id

    def log(self, metrics: dict[str, float], epoch: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=epoch)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def train(config: Config, *, reset_optimizer: bool = False) -> Path | None:
    seed_everything(config.train.seed)
    device = choose_device(config.train.device)
    resume_path = Path(config.output.resume) if config.output.resume else None
    resume = (
        torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_path else None
    )
    if resume:
        validate_resume_config(
            config, resume["config"], reset_optimizer=reset_optimizer
        )
        saved_version = resume.get("diffuser", {}).get("version")
        if saved_version != FLOW_VERSION:
            raise ValueError(
                f"Checkpoint flow version {saved_version!r} is incompatible with "
                f"{FLOW_VERSION!r}"
            )
    spur = build_spur(config, device)
    model = DirectTransformer(config.model, len(spur.class_names)).to(device)
    diffuser = MaskedFlow(
        config.spur.symmetry, float(spur.side), config.flow.kappa
    )
    start_epoch = int(resume["epoch"]) + 1 if resume else 0
    schedule_epochs = (
        config.train.num_epochs - start_epoch
        if resume and reset_optimizer
        else config.train.num_epochs
    )
    if schedule_epochs <= 0:
        raise ValueError("num_epochs must exceed the resumed checkpoint epoch")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config, schedule_epochs)
    training_generator = make_generator(device, config.train.seed)
    sampling_generator = make_generator(device, config.reverse.seed)
    if resume:
        model.load_state_dict(resume["model"])
        if not reset_optimizer:
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            _move_optimizer_state(optimizer, device)
        restore_rng(resume["rng"], training_generator, sampling_generator)
        global_step = int(resume["global_step"])
        identifier = str(resume["identifier"])
        output_directory = Path(resume["output_directory"])
        run_name = config.wandb.run_name or str(resume["wandb_run_name"])
        wandb_id = config.wandb.run_id or resume.get("wandb_run_id")
        best_epoch = int(resume["best_epoch"])
        best_loss = float(resume["best_primary_metric"])
    else:
        start_epoch, global_step, best_epoch, best_loss = 0, 0, -1, math.inf
        identifier = make_identifier(config)
        output_directory = Path(config.output.directory) / identifier
        run_name = config.wandb.run_name or make_run_name(config)
        wandb_id = config.wandb.run_id or run_name
    checkpoint_directory = output_directory / "checkpoints"
    svg_directory = output_directory / "svg"
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger = WandbLogger(
        config, run_name, wandb_id, resume is not None,
        {"total": total, "trainable": trainable},
    )
    steps = batches_per_epoch(config)
    print(f"Device: {device}\nIdentifier: {identifier}\nOutput: {output_directory}")
    last_checkpoint: Path | None = resume_path
    try:
        for epoch in range(start_epoch, config.train.num_epochs):
            started = time.perf_counter()
            model.train()
            sums = torch.zeros(4, dtype=torch.float64)
            total_sum = grad_sum = 0.0
            for _ in range(steps):
                batch = prepare_flow_batch(
                    spur,
                    diffuser,
                    config.train.batch_size,
                    training_generator,
                    config.flow.matching,
                    config.flow.match_target_size,
                    config.flow.match_max_size,
                    config.flow.lsa_workers,
                )
                prediction = model(
                    batch.state, batch.colors, batch.time, batch.labels
                )
                squared = (prediction - batch.target_velocity).square()
                loss = squared.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = clip_grad_norm_(
                    model.parameters(), config.train.grad_clip
                )
                optimizer.step()
                global_step += 1
                total_sum += loss.item()
                sums += squared.detach().mean(dim=(0, 1)).cpu().double()
                grad_sum += gradient_norm.item()
            scheduler.step()
            average = total_sum / steps
            if average < best_loss:
                best_loss, best_epoch = average, epoch
            generated, colors, _ = reverse_sample(
                model, spur, diffuser, 1, config.reverse.num_steps,
                sampling_generator,
            )
            component = sums / steps
            metrics = {
                "average_training_loss": average,
                "velocity_x_mse": component[0].item(),
                "velocity_y_mse": component[1].item(),
                "velocity_angle_mse": component[2].item(),
                "velocity_occupancy_mse": component[3].item(),
                "gradient_norm": grad_sum / steps,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "final_occupancy_error": (1 - generated[..., 3]).abs().mean().item(),
                "lattice_loss": lattice_loss(
                    config.spur.symmetry, float(spur.side),
                    generated[..., :3], colors,
                ).item(),
                "epoch_seconds": time.perf_counter() - started,
            }
            checkpoint_id = f"{identifier}_e{epoch:03d}"
            save_sample_svg(
                svg_directory / f"{checkpoint_id}.svg",
                generated[0], colors[0], config.spur.symmetry, float(spur.side),
            )
            payload = {
                "epoch": epoch, "global_step": global_step,
                "average_training_loss": average, "metrics": metrics,
                "best_epoch": best_epoch, "best_primary_metric": best_loss,
                "model": model.state_dict(),
                "diffuser": {
                    "version": FLOW_VERSION,
                    "coordinate": "quantized_polar_rank",
                    "radial_bin_width": diffuser.radial_bin_width,
                    "kappa": diffuser.kappa,
                },
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config.to_dict(), "symmetry": config.spur.symmetry,
                "side": float(spur.side), "num_tiles": config.spur.num_tiles,
                "num_ret_tiles": config.spur.num_ret_tiles,
                "num_classes": len(spur.class_names),
                "class_lookup": list(spur.class_names),
                "identifier": identifier,
                "output_directory": str(output_directory),
                "wandb_run_name": run_name,
                "wandb_run_id": logger.run_id or wandb_id,
                "rng": capture_rng(training_generator, sampling_generator),
            }
            last_checkpoint = save_epoch(
                payload, checkpoint_directory, identifier, epoch
            )
            retain_newest_and_best(
                checkpoint_directory, identifier, epoch, best_epoch
            )
            logger.log(metrics, epoch)
            print(
                f"Epoch {epoch:03d} loss={average:.6f} "
                f"occupancy_error={metrics['final_occupancy_error']:.6f} "
                f"lattice={metrics['lattice_loss']:.6f}"
            )
    finally:
        logger.finish()
    return last_checkpoint


def main() -> None:
    config, args = load_config()
    train(config, reset_optimizer=args.reset_optimizer)


if __name__ == "__main__":
    main()

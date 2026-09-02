"""PenroseSpur batches, Euler sampling, metrics, and opacity SVG output."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import re
import sys
from typing import Any

import torch

from config import Config, config_from_dict, effective_translation
from denoiser import DirectTransformer
from diffuser import MaskedFlow

PROJECT_DIR = Path(__file__).resolve().parent
SPUR_DIR = Path(
    os.environ.get("PENROSE_SPUR_PATH", PROJECT_DIR.parent / "PenroseSpur")
).resolve()
if not (SPUR_DIR / "sampler.py").exists():
    raise ImportError(f"PenroseSpur not found at {SPUR_DIR}; set PENROSE_SPUR_PATH")
sys.path.insert(0, str(SPUR_DIR))
from show import save_tiles_svg  # noqa: E402


def _load_spur_file(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"penrose_spur_{name}", SPUR_DIR / f"{name}.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load PenroseSpur module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_spur_sampler = _load_spur_file("sampler")
_spur_match = _load_spur_file("match")
_spur_lattice = _load_spur_file("lattice_loss")
SpurSampler = _spur_sampler.SpurSampler
ANGLE_SCALE = _spur_sampler.ANGLE_SCALE
lattice_loss = _spur_lattice.lattice_loss


@dataclass
class FlowBatch:
    state: torch.Tensor
    target_velocity: torch.Tensor
    colors: torch.Tensor
    labels: torch.Tensor
    time: torch.Tensor


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def build_spur(config: Config, device: torch.device) -> Any:
    return SpurSampler(
        symmetry=config.spur.symmetry,
        num_tiles=config.spur.num_tiles,
        num_ret_tiles=config.spur.num_ret_tiles,
        num_cool_classes=config.spur.num_cool_classes,
        translation_canvas=effective_translation(config.spur),
        seed=config.spur.seed,
        device=device,
        rotation_canvas=config.spur.rotation_canvas,
        rotation_mask=config.spur.rotation_mask,
    )


def prepare_flow_batch(
    spur: Any,
    diffuser: MaskedFlow,
    batch_size: int,
    generator: torch.Generator,
    matching: bool,
    match_target_size: int = 64,
    match_max_size: int = 80,
    lsa_workers: int | None = None,
) -> FlowBatch:
    batch = spur.sample_batch(batch_size, generator=generator)
    data = batch["xya"].float()
    colors, labels = batch["colors"].long(), batch["labels"].long()
    noise = spur.sample_noise(batch_size, generator=generator).to(data.dtype)
    time = diffuser.sample_training_times(batch_size, data.device, generator)
    if matching:
        noise = _spur_match.match(
            data,
            noise,
            method="lsa",
            colors=colors,
            lsa_target_size=match_target_size,
            lsa_max_size=match_max_size,
            lsa_workers=lsa_workers,
            generator=generator,
        )
    state, target = diffuser.interpolate(data, noise, time)
    return FlowBatch(state, target, colors, labels, time)


def _condition_batch(
    spur: Any,
    count: int,
    generator: torch.Generator,
    labels: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if labels is None:
        batch = spur.sample_batch(count, generator=generator)
    else:
        labels = labels.to(device=spur.device, dtype=torch.long)
        indices = []
        for label in labels:
            candidates = torch.nonzero(spur.labels == label, as_tuple=False).flatten()
            if candidates.numel() == 0:
                raise ValueError(f"Class label {int(label)} is unavailable")
            draw = torch.randint(
                candidates.numel(), (1,), device=spur.device, generator=generator
            )
            indices.append(candidates[draw])
        batch = spur.sample_batch(
            count, mask_idx=torch.cat(indices), generator=generator
        )
    return batch["colors"].long(), batch["labels"].long()


@torch.no_grad()
def reverse_sample(
    model: DirectTransformer,
    spur: Any,
    diffuser: MaskedFlow,
    count: int,
    num_steps: int,
    generator: torch.Generator,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if count <= 0:
        raise ValueError("sample count must be positive")
    colors, class_labels = _condition_batch(spur, count, generator, labels)
    xya = spur.sample_noise(count, generator=generator).float()
    state = torch.cat((xya, torch.zeros_like(xya[..., :1])), dim=-1)
    times = diffuser.sampling_times(num_steps, state.device, state.dtype)
    model.eval()
    period, half = 2.0 * (3.0 ** 0.5), 3.0 ** 0.5
    for start, stop in zip(times[:-1], times[1:]):
        velocity = model(state, colors, start.expand(count), class_labels)
        state = state + (stop - start) * velocity
        state[..., 2] = torch.remainder(state[..., 2] + half, period) - half
        state[..., 3].clamp_(0.0, 1.0)
    return state, colors, class_labels


def save_sample_svg(
    path: Path,
    state: torch.Tensor,
    colors: torch.Tensor,
    symmetry: int,
    side: float,
) -> Path:
    opacity = 0.15 + 0.60 * state[..., 3].clamp(0.0, 1.0)
    return save_tiles_svg(
        path,
        state[..., :3],
        colors,
        symmetry=symmetry,
        side=side,
        angle_scale=ANGLE_SCALE,
        background="#000000",
        show_arcs=symmetry == 5,
        opacities=opacity,
        alpha=1.0,
    )


def resolve_class_label(spur: Any, value: str) -> int:
    try:
        label = int(value)
    except ValueError:
        matches = [
            index for index, name in enumerate(spur.class_names)
            if name.casefold() == value.casefold()
        ]
        if not matches:
            raise ValueError(f"Unknown class name: {value}") from None
        label = matches[0]
    if not 0 <= label < len(spur.class_names) or not bool((spur.labels == label).any()):
        raise ValueError(f"Class label {label} is not in the configured subset")
    return label


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-") or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample PenroseFoam with Euler")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("-n", "--count", type=int)
    parser.add_argument("-r", "--seed", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("-c", "--class-name")
    parser.add_argument("--symmetry", type=int, choices=(5, 6))
    parser.add_argument("--num-tiles", type=int)
    parser.add_argument("-d", "--device", default="auto")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    values = checkpoint["config"]
    if args.symmetry is not None:
        values["spur"]["symmetry"] = args.symmetry
    if args.num_tiles is not None:
        values["spur"]["num_tiles"] = values["spur"]["num_ret_tiles"] = args.num_tiles
    config = config_from_dict(values)
    device = choose_device(args.device)
    spur = build_spur(config, device)
    model = DirectTransformer(config.model, len(spur.class_names)).to(device)
    model.load_state_dict(checkpoint["model"])
    diffuser = MaskedFlow(config.flow.kappa)
    count = args.count or config.reverse.n
    seed = config.reverse.seed if args.seed is None else args.seed
    steps = args.num_steps or config.reverse.num_steps
    generator = make_generator(device, seed)
    labels = None
    if args.class_name is not None:
        label = resolve_class_label(spur, args.class_name)
        labels = torch.full((count,), label, device=device, dtype=torch.long)
    state, colors, class_labels = reverse_sample(
        model, spur, diffuser, count, steps, generator, labels
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        name = _safe_name(spur.class_names[int(class_labels[index])])
        save_sample_svg(
            args.output / f"sample_{index:04d}_{name}_s{seed + index}.svg",
            state[index], colors[index], config.spur.symmetry, float(spur.side)
        )
    print(f"Saved {count} SVG samples to {args.output}")


if __name__ == "__main__":
    main()

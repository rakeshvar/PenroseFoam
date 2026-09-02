"""Strict typed configuration for PenroseFoam."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, TypeVar

import torch
import yaml

PROJECT_DIR = Path(__file__).resolve().parent
T = TypeVar("T")


@dataclass
class SpurConfig:
    symmetry: int | None = None
    num_tiles: int = 96
    num_ret_tiles: int = 96
    num_cool_classes: int | None = 10
    translation_canvas: float | None = None
    seed: int | None = None
    rotation_canvas: float = math.pi
    rotation_mask: float = math.pi / 4


@dataclass
class ModelConfig:
    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 8
    num_global_tokens: int = 8
    class_embed_dim: int = 32
    time_embed_dim: int = 32
    dropout: float = 0.0


@dataclass
class FlowConfig:
    sigma: float = 1.0
    kappa: float = 0.05
    matching: bool = True
    match_target_size: int = 64
    match_max_size: int = 80
    lsa_workers: int = 8


@dataclass
class TrainConfig:
    batch_size: int = 64
    samples_per_class: int = 1920
    samples_per_epoch: int | None = None
    num_epochs: int = 101
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    min_lr_factor: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cuda"


@dataclass
class ReverseConfig:
    n: int = 1
    seed: int = 1
    num_steps: int = 100


@dataclass
class WandbConfig:
    enable: bool = True
    project: str = "penrose-foam"
    run_name: str | None = None
    run_id: str | None = None


@dataclass
class OutputConfig:
    directory: str = "outputs"
    resume: str | None = None


@dataclass
class Config:
    spur: SpurConfig
    model: ModelConfig
    flow: FlowConfig
    train: TrainConfig
    reverse: ReverseConfig
    wandb: WandbConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SECTIONS: dict[str, type[Any]] = {
    "spur": SpurConfig, "model": ModelConfig, "flow": FlowConfig,
    "train": TrainConfig, "reverse": ReverseConfig, "wandb": WandbConfig,
    "output": OutputConfig,
}
IMMUTABLE_ON_RESUME = (
    "spur.symmetry", "spur.num_tiles", "spur.num_ret_tiles",
    "spur.num_cool_classes", "spur.translation_canvas", "spur.seed",
    "spur.rotation_canvas", "spur.rotation_mask", "model.d_model",
    "model.num_heads", "model.num_layers", "model.num_global_tokens",
    "model.class_embed_dim", "model.time_embed_dim", "model.dropout",
    "flow.sigma", "flow.kappa", "flow.matching", "flow.match_target_size",
    "flow.match_max_size", "flow.lsa_workers", "train.batch_size",
    "train.samples_per_epoch", "train.learning_rate", "train.weight_decay",
    "train.min_lr_factor", "train.grad_clip", "train.seed",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value


def _merge(target: dict[str, Any], source: dict[str, Any], prefix: str = "") -> None:
    for key, value in source.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if key not in target:
            raise ValueError(f"Unknown configuration key: {dotted}")
        if isinstance(target[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected a mapping for {dotted}")
            _merge(target[key], value, dotted)
        else:
            target[key] = value


def _override(target: dict[str, Any], section: str | None, expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must be key=value, got {expression!r}")
    dotted, raw = expression.split("=", 1)
    if section is not None and "." not in dotted:
        dotted = f"{section}.{dotted}"
    keys, cursor = dotted.split("."), target
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            raise ValueError(f"Unknown configuration key: {dotted}")
        cursor = cursor[key]
    if keys[-1] not in cursor:
        raise ValueError(f"Unknown configuration key: {dotted}")
    cursor[keys[-1]] = yaml.safe_load(raw)


def _make_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    unknown = set(values) - {field.name for field in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def config_from_dict(values: dict[str, Any]) -> Config:
    missing, unknown = set(SECTIONS) - set(values), set(values) - set(SECTIONS)
    if missing or unknown:
        raise ValueError(
            f"Configuration sections mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    config = Config(**{
        name: _make_dataclass(kind, copy.deepcopy(values[name]))
        for name, kind in SECTIONS.items()
    })
    validate(config)
    return config


def validate(config: Config) -> None:
    if config.spur.symmetry not in (5, 6):
        raise ValueError("spur.symmetry is required and must be 5 or 6")
    if min(config.spur.num_tiles, config.spur.num_ret_tiles) <= 1:
        raise ValueError("tile counts must exceed one")
    if config.spur.num_cool_classes is not None and not 1 <= config.spur.num_cool_classes <= 30:
        raise ValueError("spur.num_cool_classes must be null or in [1,30]")
    model = config.model
    if model.d_model <= 0 or model.d_model % model.num_heads:
        raise ValueError("model.d_model must be positive and divisible by num_heads")
    if min(model.num_heads, model.num_layers, model.num_global_tokens,
           model.class_embed_dim, model.time_embed_dim) <= 0:
        raise ValueError("model dimensions and counts must be positive")
    if model.time_embed_dim < 4 or model.time_embed_dim % 2:
        raise ValueError("model.time_embed_dim must be even and at least 4")
    if not 0 <= model.dropout < 1:
        raise ValueError("model.dropout must be in [0,1)")
    flow = config.flow
    if min(flow.sigma, flow.kappa) <= 0:
        raise ValueError("flow sigma and kappa must be positive")
    if min(flow.match_target_size, flow.match_max_size, flow.lsa_workers) <= 0:
        raise ValueError("matching sizes and workers must be positive")
    if flow.match_target_size > flow.match_max_size:
        raise ValueError("match_target_size cannot exceed match_max_size")
    if min(config.train.batch_size, config.train.samples_per_class,
           config.train.num_epochs, config.reverse.n, config.reverse.num_steps) <= 0:
        raise ValueError("all count settings must be positive")
    if config.train.samples_per_epoch is not None and config.train.samples_per_epoch <= 0:
        raise ValueError("train.samples_per_epoch must be positive or null")
    if config.train.learning_rate <= 0 or config.train.grad_clip <= 0:
        raise ValueError("learning rate and gradient clip must be positive")
    if config.train.weight_decay < 0 or not 0 < config.train.min_lr_factor <= 1:
        raise ValueError("invalid optimizer settings")


def effective_translation(config: SpurConfig) -> float:
    if config.translation_canvas is not None:
        return float(config.translation_canvas)
    return math.sqrt(config.num_tiles) if config.symmetry == 5 else 2.0


def effective_samples_per_epoch(config: Config) -> int:
    return config.train.samples_per_epoch or (
        config.train.samples_per_class * (config.spur.num_cool_classes or 70)
    )


def batches_per_epoch(config: Config) -> int:
    return math.ceil(effective_samples_per_epoch(config) / config.train.batch_size)


def symmetry_name(symmetry: int | None) -> str:
    return "hex" if symmetry == 6 else "pen"


def architecture_name(config: Config) -> str:
    return f"{config.model.d_model}x{config.model.num_layers}"


def make_identifier(config: Config, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return (
        f"foam_{now:%m%d_%H%M}_{symmetry_name(config.spur.symmetry)}"
        f"{config.spur.num_ret_tiles}_{architecture_name(config)}_mse_cc10"
    )


def make_run_name(config: Config) -> str:
    return (
        f"foam-{symmetry_name(config.spur.symmetry)}{config.spur.num_ret_tiles}-"
        f"{architecture_name(config)}-mse-cc10"
    )


def nested_value(mapping: dict[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for key in dotted.split("."):
        value = value[key]
    return value


def validate_resume_config(config: Config, saved: dict[str, Any]) -> None:
    current = config.to_dict()
    changed = [key for key in IMMUTABLE_ON_RESUME
               if nested_value(current, key) != nested_value(saved, key)]
    if changed:
        raise ValueError("Immutable resume configuration changed: " + ", ".join(changed))


def load_config(argv: list[str] | None = None) -> tuple[Config, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Train PenroseFoam masked flow")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--symmetry", type=int, choices=(5, 6))
    parser.add_argument("--num-tiles", type=int)
    parser.add_argument("--num-cool-classes", type=int)
    parser.add_argument("--translation", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--reverse-steps", type=int)
    parser.add_argument("--output")
    parser.add_argument("--n", type=int)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-name")
    parser.add_argument("-t", "--train", action="append", metavar="KEY=VALUE")
    parser.add_argument("-m", "--model", action="append", metavar="KEY=VALUE")
    parser.add_argument("-p", "--spur", action="append", metavar="KEY=VALUE")
    parser.add_argument("-f", "--flow", action="append", metavar="KEY=VALUE")
    parser.add_argument("-r", "--reverse", action="append", metavar="KEY=VALUE")
    parser.add_argument("-w", "--wandb", action="append", metavar="KEY=VALUE")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    merged = _read_yaml(PROJECT_DIR / "config.yaml")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        _merge(merged, checkpoint["config"])
    if args.config:
        _merge(merged, _read_yaml(args.config))
    for section, expressions in (
        ("train", args.train), ("model", args.model), ("spur", args.spur),
        ("flow", args.flow), ("reverse", args.reverse), ("wandb", args.wandb),
    ):
        for expression in expressions or []:
            _override(merged, section, expression)
    for expression in args.overrides:
        _override(merged, None, expression)
    conveniences = {
        ("spur", "symmetry"): args.symmetry,
        ("spur", "num_tiles"): args.num_tiles,
        ("spur", "num_cool_classes"): args.num_cool_classes,
        ("spur", "translation_canvas"): args.translation,
        ("train", "batch_size"): args.batch_size,
        ("train", "learning_rate"): args.learning_rate,
        ("reverse", "num_steps"): args.reverse_steps,
        ("reverse", "n"): args.n,
        ("output", "directory"): args.output,
        ("wandb", "project"): args.wandb_project,
        ("wandb", "run_name"): args.wandb_name,
    }
    for (section, key), value in conveniences.items():
        if value is not None:
            merged[section][key] = value
    if args.num_tiles is not None:
        merged["spur"]["num_ret_tiles"] = args.num_tiles
    if args.resume:
        merged["output"]["resume"] = str(args.resume)
    config = config_from_dict(merged)
    print(yaml.safe_dump(config.to_dict(), sort_keys=False).rstrip())
    return config, args

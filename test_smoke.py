"""End-to-end local smoke checks for PenroseFoam."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import torch
import yaml

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from checkpoint import retain_newest_and_best  # noqa: E402
from config import config_from_dict, load_config, make_identifier  # noqa: E402
from denoiser import DirectTransformer  # noqa: E402
from diffuser import FLOW_VERSION, MaskedFlow, canonicalize_anchor_frame  # noqa: E402
from match import balanced_group_sizes, match  # noqa: E402
from sampler import (  # noqa: E402
    build_spur, make_generator, prepare_flow_batch, reverse_sample, save_sample_svg,
)


def smoke_values(output: Path, epochs: int = 1) -> dict:
    with (PROJECT_DIR / "config.yaml").open(encoding="utf-8") as handle:
        values = copy.deepcopy(yaml.safe_load(handle))
    values["spur"].update(
        symmetry=6, num_tiles=12, num_ret_tiles=12,
        translation_canvas=2.0, seed=7,
    )
    values["model"].update(
        d_model=16, num_heads=4, num_layers=1, num_global_tokens=2,
        class_embed_dim=4, time_embed_dim=4,
    )
    values["flow"].update(lsa_workers=1)
    values["train"].update(
        batch_size=2, samples_per_epoch=2, num_epochs=epochs,
        device="cpu", seed=11,
    )
    values["reverse"].update(n=1, seed=13, num_steps=2)
    values["wandb"]["enable"] = False
    values["output"]["directory"] = str(output)
    return values


def check_configs_and_model(root: Path) -> None:
    for filename, tiles, width, layers, globals_ in (
        ("config.yaml", 96, 128, 8, 8),
        ("config384.yaml", 384, 256, 12, 16),
    ):
        config, _ = load_config(["--config", str(PROJECT_DIR / filename), "--symmetry", "6"])
        assert config.spur.num_tiles == tiles
        assert config.train.batch_size == 64
        assert config.model.d_model == width
        assert config.model.num_layers == layers
        assert config.model.num_heads == 8
        assert config.model.num_global_tokens == globals_
        assert config.model.class_embed_dim == config.model.time_embed_dim == 32
        assert config.flow.matching and config.reverse.num_steps == 100
    try:
        load_config([])
    except ValueError as error:
        assert "symmetry is required" in str(error)
    else:
        raise AssertionError("null symmetry was accepted")

    config = config_from_dict(smoke_values(root))
    fixed = datetime(2026, 9, 1, 12, 34, tzinfo=timezone.utc)
    assert make_identifier(config, fixed) == "foam_0901_1234_hex12_16x1_mse_cc10"
    model = DirectTransformer(config.model, 70)
    state = torch.randn(2, 12, 4)
    output = model(
        state, torch.randint(0, 2, (2, 12)), torch.rand(2),
        torch.randint(0, 70, (2,)),
    )
    assert output.shape == state.shape and torch.all(output[..., 3] >= 0)
    assert torch.equal(output[:, 0], torch.zeros_like(output[:, 0]))

    values384 = yaml.safe_load((PROJECT_DIR / "config384.yaml").read_text())
    values384["spur"]["symmetry"] = 6
    model384 = DirectTransformer(config_from_dict(values384).model, 70)
    with torch.no_grad():
        result = model384(
            torch.randn(1, 4, 4), torch.randint(0, 2, (1, 4)),
            torch.rand(1), torch.randint(0, 70, (1,)),
        )
    assert result.shape == (1, 4, 4) and torch.isfinite(result).all()


def check_masked_flow() -> None:
    flow = MaskedFlow(sigma=1.0, kappa=0.05)
    raw = torch.tensor([[[2.0, 1.0, 1.0], [0.1, 0.1, 0.5]]])
    data, order, anchor_angle = canonicalize_anchor_frame(raw)
    assert order.tolist() == [[1, 0]]
    assert torch.equal(data[:, 0], torch.zeros_like(data[:, 0]))
    assert torch.allclose(anchor_angle, torch.tensor([0.5]))
    data = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, -1.70]]])
    noise = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    rho = flow.radius_coordinate(data)
    assert rho[0, 0] < rho[0, 1]
    endpoints = torch.tensor([0.0, 1.0])
    duplicated_rho = rho.expand(2, -1)
    u, du = flow.occupancy(endpoints, duplicated_rho)
    assert torch.equal(u[0], torch.zeros_like(u[0]))
    assert torch.equal(u[1], torch.ones_like(u[1]))
    assert torch.isfinite(du).all() and torch.all(du >= 0)
    middle, _ = flow.occupancy(torch.tensor([0.5]), rho)
    assert middle[0, 0] > middle[0, 1]
    state, velocity = flow.interpolate(data, noise, torch.tensor([0.5]))
    assert torch.isfinite(state).all() and torch.isfinite(velocity).all()
    assert torch.equal(state[:, 0], torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    assert torch.equal(velocity[:, 0], torch.zeros_like(velocity[:, 0]))
    assert state[0, 1, 2] < 0 and velocity[0, 1, 2] < 0
    assert torch.all(velocity[..., 3] >= 0)


def check_balanced_matching() -> None:
    for count, expected in (
        (40, (40,)), (80, (80,)), (81, (41, 40)),
        (96, (48, 48)), (128, (64, 64)), (160, (54, 53, 53)),
        (161, (54, 54, 53)), (237, (60, 59, 59, 59)),
        (256, (64, 64, 64, 64)),
    ):
        sizes = balanced_group_sizes(count)
        assert sizes == expected and max(sizes) <= 80 and max(sizes) - min(sizes) <= 1
        assert sum(sizes) == count
    for count in range(1, 513):
        sizes = balanced_group_sizes(count)
        assert sum(sizes) == count
        assert max(sizes) <= 80 and max(sizes) - min(sizes) <= 1
        valid_over_40 = math.ceil(count / 80) <= count // 41
        if valid_over_40:
            assert min(sizes) > 40
    count = 161
    data, noise = torch.randn(1, count, 3), torch.randn(1, count, 3)
    colors = torch.tensor([[0] * 81 + [1] * 80])
    matched = match(
        data, noise, method="lsa", colors=colors, lsa_workers=1,
        generator=torch.Generator().manual_seed(23),
    )
    repeated = match(
        data, noise, method="lsa", colors=colors, lsa_workers=1,
        generator=torch.Generator().manual_seed(23),
    )
    assert torch.equal(matched, repeated)
    for color in (0, 1):
        before = noise[0, colors[0] == color].sort(dim=0).values
        after = matched[0, colors[0] == color].sort(dim=0).values
        assert torch.allclose(before, after)


def check_optimizer_euler_svg(root: Path) -> None:
    config = config_from_dict(smoke_values(root))
    spur = build_spur(config, torch.device("cpu"))
    generator = make_generator(torch.device("cpu"), 17)
    flow = MaskedFlow(config.flow.sigma, config.flow.kappa)
    batch = prepare_flow_batch(spur, flow, 2, generator, True, 64, 80, 1)
    model = DirectTransformer(config.model, len(spur.class_names))
    prediction = model(batch.state, batch.colors, batch.time, batch.labels)
    loss = (prediction - batch.target_velocity).square().mean()
    assert torch.isfinite(loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    generated, colors, _ = reverse_sample(model, spur, flow, 1, 2, generator)
    assert generated.shape == (1, 12, 4)
    assert torch.isfinite(generated).all()
    assert torch.all((0 <= generated[..., 3]) & (generated[..., 3] <= 1))
    assert torch.equal(generated[:, 0, :2], torch.zeros_like(generated[:, 0, :2]))
    assert torch.equal(generated[:, 0, 3], torch.ones_like(generated[:, 0, 3]))
    generated[0, :, 3] = torch.linspace(0, 1, 12)
    path = save_sample_svg(root / "opacity.svg", generated[0], colors[0], 6, float(spur.side))
    polygons = ET.parse(path).findall(".//{http://www.w3.org/2000/svg}polygon")
    assert len(polygons) == 12
    opacity = torch.tensor([float(item.attrib["opacity"]) for item in polygons])
    assert torch.allclose(opacity, generated[0, :, 3], atol=5e-5)


def check_retention(root: Path) -> None:
    root.mkdir(parents=True)
    for epoch in range(4):
        (root / f"foam_test_e{epoch:03d}.pt").touch()
    retain_newest_and_best(root, "foam_test", 3, 1)
    assert {path.name for path in root.glob("*.pt")} == {
        "foam_test_e001.pt", "foam_test_e003.pt",
    }


def run_train(arguments: list[str], environment: dict[str, str]) -> None:
    result = subprocess.run(
        [sys.executable, "-u", "train.py", *arguments],
        cwd=PROJECT_DIR, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise AssertionError(result.stdout)


def newest_checkpoint(output: Path) -> tuple[Path, dict]:
    paths = list(output.glob("*/checkpoints/*_e*.pt"))
    assert paths
    loaded = [(path, torch.load(path, map_location="cpu", weights_only=False)) for path in paths]
    return max(loaded, key=lambda item: item[1]["epoch"])


def check_fresh_process_resume(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PENROSE_SPUR_PATH"] = str(PROJECT_DIR.parent / "PenroseSpur")
    output = root / "outputs"
    first_config = root / "first.yaml"
    first_config.write_text(yaml.safe_dump(smoke_values(output, 1), sort_keys=False))
    run_train(["--config", str(first_config)], environment)
    first_path, first = newest_checkpoint(output)
    assert first["epoch"] == 0 and first["global_step"] == 1
    assert first["diffuser"] == {
        "version": FLOW_VERSION,
        "sigma": 1.0,
        "kappa": 0.05,
    }
    assert first["identifier"].startswith("foam_")
    assert set(first["rng"]) == {
        "python", "numpy", "torch_cpu", "torch_cuda",
        "training_generator", "sampling_generator",
    }
    run_train(["--resume", str(first_path), "-t", "num_epochs=2"], environment)
    _, resumed = newest_checkpoint(output)
    assert resumed["epoch"] == 1 and resumed["global_step"] == 2
    assert resumed["identifier"] == first["identifier"]
    assert resumed["output_directory"] == first["output_directory"]
    svg = Path(first["output_directory"]) / "svg"
    assert (svg / f"{first['identifier']}_e000.svg").exists()
    assert (svg / f"{first['identifier']}_e001.svg").exists()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="penrose-foam-smoke-") as temporary:
        root = Path(temporary)
        check_configs_and_model(root / "config")
        check_masked_flow()
        check_balanced_matching()
        check_optimizer_euler_svg(root / "flow")
        check_retention(root / "retention")
        check_fresh_process_resume(root / "resume")
    print("PenroseFoam smoke test passed")


if __name__ == "__main__":
    main()

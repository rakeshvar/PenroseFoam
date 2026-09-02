# PenroseFoam

PenroseFoam is an independent PyTorch experiment for class-conditioned masked
optimal-transport flow on live PenroseSpur batches. Its only state is
`(x, y, scaled_angle, occupancy)` and its only sampler is 100-step Euler.

For each clean tile, radius is mapped to the Rayleigh CDF coordinate
`rho = 1 - exp(-r^2 / (2 sigma^2))`. An endpoint-normalized logistic with
`kappa=0.05` gives exact occupancy zero at time zero and one at time one.
Its nonnegative analytic derivative multiplies the shortest XYA endpoint
displacement to produce the joint four-channel velocity target. The model's
occupancy velocity uses softplus, so sampled occupancy cannot decrease.

PenroseSpur must be a sibling checkout, or `PENROSE_SPUR_PATH` must point to it.
PenroseFoam has no runtime dependency on any other Penrose project.

## Configurations

Both production configurations intentionally leave `spur.symmetry: null`.
Every train launch must explicitly choose `--symmetry 6` (hex) or
`--symmetry 5` (Penrose).

- `config.yaml`: 96 tiles, 128-wide, 8 layers, 8 heads, 8 global tokens.
- `config384.yaml`: 384 tiles, 256-wide, 12 layers, 8 heads, 16 global tokens.
- Both use 32-dimensional class/time embeddings, batch 64, the first 10 cool
  MPEG7 classes, matching enabled, and 100 Euler steps.

Matching uses PenroseSpur's shared default LSA API: exact same-color solves
within independently shuffled, balanced groups. Groups target 64 tiles, never
exceed 80, differ in size by at most one within each color, and only contain 40
tiles or fewer when no all-above-40 partition is possible. Foam passes its
training generator into Spur, so checkpoint resume reproduces the grouping.

Actual batch-64 memory viability for the 384-tile model depends on the target
GPU.

## Train

```bash
PENROSE_SPUR_PATH=../PenroseSpur ~/.aivenv/bin/python train.py \
  --config config.yaml --symmetry 6

PENROSE_SPUR_PATH=../PenroseSpur ~/.aivenv/bin/python train.py \
  --config config384.yaml --symmetry 5
```

Known section overrides can be repeated, for example
`-t batch_size=32 -f matching=false -r num_steps=50`. Convenience flags cover
symmetry, tile count, class count, translation, batch size, learning rate,
reverse steps, sample count, output, and WandB naming. Unknown keys fail.

Run directories use
`foam_<MMDD>_<HHMM>_<hex|pen><96|384>_<width>x<layers>_mse_cc10`.
Each epoch writes a matching SVG and atomic checkpoint. Only newest and
best-average-MSE checkpoints remain.

## Resume

Resume preserves identifier, output location, optimizer, scheduler, WandB
identity, epoch numbering, and Python/NumPy/PyTorch/Spur generator states:

```bash
PENROSE_SPUR_PATH=../PenroseSpur ~/.aivenv/bin/python train.py \
  --resume outputs/foam_0901_1200_hex96_128x8_mse_cc10/checkpoints/foam_0901_1200_hex96_128x8_mse_cc10_e100.pt \
  -t num_epochs=201
```

Model, Spur, flow, matching, and optimization settings are immutable on resume.

## Sample

```bash
PENROSE_SPUR_PATH=../PenroseSpur ~/.aivenv/bin/python sampler.py \
  CHECKPOINT.pt -n 4 -r 7 -c apple --num-steps 100 \
  --symmetry 6 --num-tiles 96 -o samples
```

The class can be a configured MPEG7 name or numeric ID. Every generated tile
is rendered; its clamped predicted occupancy is used directly as SVG opacity.

## Test

```bash
PENROSE_SPUR_PATH=../PenroseSpur ~/.aivenv/bin/python test_smoke.py
```

The smoke suite covers both production configs, exact occupancy endpoints,
radius ordering, analytic velocities, periodic angle paths, balanced grouped
same-color matching at high time, a finite optimizer step, Euler integration,
SVG opacity, checkpoint retention, and fresh-process resume.

## Deliberate boundaries

There is no alternate schedule, sampler, matcher, loss, denoiser, method
registry, lattice training term, pre-generated dataset, `.npz` path,
dataset-generation code, XLA/JAX/TPU support, or cloud launcher. WandB receives
scalar metrics only.

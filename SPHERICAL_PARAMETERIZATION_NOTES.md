# Progressive bijective spherical parameterization

This repository now includes `spherical_parameterization.py`, which implements a
flip-safe progressive spherical map optimizer for genus-0 triangle meshes.

## Why this works better for elongated parts (legs/tails)

The algorithm addresses the two failure modes you described:

1. **Triangle flips / overlap** are blocked by strict feasibility checks.
   Every accepted update must keep every spherical face positively oriented
   with area above a threshold (`min_signed_area`).
2. **Local fold pressure** is reduced with per-step re-optimization after each
   vertex insertion, plus optional repulsion and edge-length regularization.

## Core API

- `ProgressiveStep`: one reverse-collapse insertion event.
- `SphericalParams`: optimization hyperparameters.
- `ProgressiveSphericalParameterizer.run_progressive(...)`: main driver.

## Practical settings

Start from these values for noisy anatomical meshes:

- `max_iter_per_step = 120`
- `smooth_weight = 1.0`
- `edge_weight = 0.1`
- `repel_weight = 0.002`
- `min_signed_area = 5e-7`

If you still see near-folding (tiny positive areas), increase `edge_weight`
slightly and reduce insertion batch size so each step is less aggressive.

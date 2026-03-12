# Audit and Upgrade Plan (E2E-CDRO Demo)

## What was wrong in the previous implementation

1. **Baseline mismatch**
   - Main scripts compared `mpc/static_cdro/ddro/e2e_cdro/ppo` rather than the required set.
   - PPO and legacy variants were mixed into headline outputs.

2. **Plant oversimplification**
   - Main demo used a scalar single-zone state with one disturbance scalar.
   - No spatial temperature effects, no hotspot migration, and no explicit coolant pathway states.

3. **Experiment too lightweight**
   - Short trajectories and narrow scenario diversity.
   - No proper ID/OOD split and weak multi-seed aggregation.

4. **Visualization quality gap**
   - Minimal default plots, no thermal schematics, no chip maps, no space-time visuals.

## Upgrades implemented

- **New main plant**: `GridCoolingEnv` (8x8 chip grid + cold plate + coolant states + pump actuation and slew limits).
- **Required main baseline set only**:
  1) PID
  2) Deterministic MPC
  3) Robust MPC
  4) Non-contextual DRO
  5) Contextual DRO (E2E-CDRO-style context-adaptive robustness)
- **Main evaluation rewrite**:
  - ID/OOD scenario families, longer episodes, multi-seed runs, aggregate mean/std metrics.
- **Visualization overhaul**:
  - Pipeline schematic, thermal plant schematic, multi-panel time series, chip snapshots, space-time map, coolant-flow figure, summary bars.

## Main showcase vs debug case

- **Main showcase**: grid + liquid-cooling surrogate (`plant_kind: grid`).
- **Debug/testing case**: scalar plant is preserved (`plant_kind: scalar`) for legacy tests and sanity checks.

## Remaining approximations (explicit)

- Robust/DRO optimization uses a reduced-order surrogate over hotspot/coolant states rather than full high-dimensional MPC.
- Contextual DRO currently uses a context-conditioned radius mapping (configurable) and is structured for later refinement.
- FMU path remains available as an optional backend and does not block main Python-based experiments.

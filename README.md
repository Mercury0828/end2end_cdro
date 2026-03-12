# E2E-CDRO Thermal Control Demo (Research-Grade Upgrade)

This repository now uses a **liquid-cooling-oriented thermal benchmark** as the main showcase:

- 2D chip thermal grid (default 8x8)
- cold-plate lumped temperature
- coolant lumped temperature
- pump-speed control with saturation + slew constraints
- spatial workload maps with burst, drift, plateau, and combined stress regimes

## Main comparison set (non-negotiable)

The primary experiment suite reports **exactly**:
1. PID
2. Deterministic MPC
3. Robust MPC
4. Non-contextual DRO
5. Contextual DRO / E2E-CDRO

Legacy controllers remain in source for backward compatibility but are excluded from the main evaluation and plots.

## Project structure

- `src/plant/grid_cooling_env.py`: main plant (chip + cold plate + coolant)
- `src/scenario/thermal_workload.py`: richer ID/OOD workload and context generation
- `src/controllers/main_suite.py`: required controller suite + PID tuning logic
- `scripts/eval_all.py`: multi-seed evaluation, train/val/test-style splits, aggregate metrics
- `scripts/make_plots.py`: presentation-grade figure generation
- `AUDIT_AND_UPGRADE_PLAN.md`: audit findings and redesign notes

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run main experiment

```bash
python scripts/eval_all.py --base configs/base.yaml --out outputs/main_run
python scripts/make_plots.py --results outputs/main_run/results_per_step.csv --summary outputs/main_run/summary.csv --outdir outputs/main_run/figures
```

## Run one method demo

```bash
python scripts/run_single_demo.py --controller contextual_dro
```

## FMU backend support

FMU support remains available via config switch:

```yaml
plant:
  mode: fmu
  fmu_path: artifacts/SingleZoneThermal.fmu
```

If FMU export/runtime is unavailable, the main suite remains runnable in Python mode.

## Debug scalar mode (for tests/sanity only)

```yaml
plant:
  plant_kind: scalar
```

This is intentionally demoted and not used for the main reported results.

## Testing

```bash
pytest -q
```

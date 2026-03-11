# E2E-CDRO Demo (Modelica -> FMU -> FMPy -> Python orchestration)

This repository provides a runnable end-to-end demo for the synthetic mechanism verification stage:

- Plant: first-order single-zone thermal model
- Toolchain: Modelica/OpenModelica FMU export + FMPy runtime + Python fallback
- Controllers: MPC, Static-CDRO, D-DRO, E2E-CDRO, PPO hook
- Outputs: metrics tables + presentation-ready plots + MLflow logs

## Adopted assumptions and explicit approximations

Because the environment may not always have PDF parsing tools and FMU toolchains installed, this implementation follows the synthetic setup specified in your request and keeps every approximation explicit:

1. Dynamics implemented as requested:
   \(x_{t+1}=x_t+\alpha(Tamb_t-x_t)-\beta u_t+\gamma \xi_t+\epsilon_t\).
2. Robust lower layer approximation:
   - Convex single-step surrogate with worst-case disturbance `xi_nominal + rho`.
   - Slack `s >= x_next` used for `max(0, x_next)` safety penalty.
3. E2E training approximation:
   - First working path uses a lightweight supervised/proxy target for proactive `rho(z)` behavior.
   - Code is modular so implicit-KKT/SOCP can be swapped in later under `src/optimization` and `src/learning`.
4. FMU fallback:
   - If FMU or FMPy is unavailable, the same interface uses the pure-Python plant.

## Project layout

See `configs/`, `modelica/`, `src/`, `scripts/`, and `tests/` for the full modular pipeline.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow commands

### 1) Export FMU (OpenModelica)

```bash
python scripts/export_fmu.py
# or
cd modelica && omc export_fmu.mos
```

### 2) Run one trajectory demo

```bash
python scripts/run_single_demo.py --controller mpc
python scripts/run_single_demo.py --controller static_cdro
python scripts/run_single_demo.py --controller ddro
python scripts/run_single_demo.py --controller e2e_cdro
```

### 3) Train E2E-CDRO

```bash
python scripts/train_e2e.py
```

### 4) Evaluate all controllers on held-out trajectories

```bash
python scripts/eval_all.py --out outputs/default_run
```

### 5) Generate scenario/mechanism/outcome plots

```bash
python scripts/make_plots.py --results outputs/default_run/results_per_step.csv --outdir outputs/default_run
```

## FMU vs Python mode

Switch `configs/base.yaml`:

```yaml
plant:
  mode: python  # set to fmu to prefer FMU runtime
  fmu_path: artifacts/SingleZoneThermal.fmu
```

## Testing

```bash
pytest -q
```

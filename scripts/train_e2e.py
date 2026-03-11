#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import torch

from src.utils.config import load_yaml
from src.utils.seeds import set_seed
from src.utils.io import ensure_dir
from src.scenario.calm_burst_calm import ScenarioConfig, generate_dataset
from src.controllers.e2e_cdro import E2ECDROController
from src.learning.trainer import train_e2e
from src.logging_utils.mlflow_logger import MLflowLogger

if __name__ == "__main__":
    base = load_yaml("configs/base.yaml")
    ecfg = load_yaml("configs/e2e_cdro.yaml")
    set_seed(base["seed"])
    scfg = ScenarioConfig(**load_yaml("configs/scenario.yaml"))
    train_set = generate_dataset(scfg, base["evaluation"]["n_train"], seed=base["seed"])

    p = base["plant"]
    ctrl = E2ECDROController(p["alpha"], p["beta"], p["gamma"], 10.0, base["lambda_penalty"], hidden_dim=ecfg["model"]["hidden_dim"], lr=ecfg["train"]["lr"])
    losses = train_e2e(ctrl, train_set, epochs=ecfg["train"]["epochs"])

    ckpt_dir = ensure_dir("artifacts/checkpoints")
    ckpt = Path(ckpt_dir) / "e2e_rho_net.pt"
    torch.save(ctrl.net.state_dict(), ckpt)

    mlf = MLflowLogger()
    with mlf.run("train_e2e_cdro"):
        mlf.log_config({"base": base, "e2e": ecfg})
        for i, loss in enumerate(losses):
            mlf.log_metrics({"train_loss": loss}, step=i)
        mlf.log_artifact(str(ckpt))

    print(f"Saved checkpoint: {ckpt}")

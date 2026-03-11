from __future__ import annotations

from pathlib import Path
import mlflow


class MLflowLogger:
    def __init__(self, experiment="e2e_cdro_demo"):
        mlflow.set_experiment(experiment)

    def run(self, run_name: str):
        return mlflow.start_run(run_name=run_name)

    @staticmethod
    def log_config(cfg: dict):
        mlflow.log_dict(cfg, "config.yaml")

    @staticmethod
    def log_metrics(metrics: dict, step: int | None = None):
        mlflow.log_metrics(metrics, step=step)

    @staticmethod
    def log_artifact(path: str):
        if Path(path).exists():
            mlflow.log_artifact(path)

from __future__ import annotations

import pandas as pd


def summarize_runs(per_episode_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in per_episode_df.columns if c not in ["controller", "episode"]]
    return per_episode_df.groupby("controller")[metric_cols].mean().reset_index()

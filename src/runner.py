from __future__ import annotations

import pandas as pd
from src.metrics.metrics import compute_episode_metrics


def rollout(env, controller, scenario, lambda_penalty=100.0, safety_threshold=0.0):
    controller.reset()
    env.reset()
    rows = []
    infeasible = 0
    solve_times = []
    for t in scenario["t"]:
        x = env.get_state()
        z = float(scenario["z"][t])
        xi = float(scenario["xi"][t])
        tamb = float(scenario["Tamb"][t])
        if hasattr(controller, "observe"):
            controller.observe(xi, z)
        u, rho, rob = controller.act(x, z, tamb)
        x_next = env.step(u, tamb, xi)
        cost = u + lambda_penalty * max(0.0, x_next - safety_threshold)
        status = getattr(rob, "status", "na") if rob is not None else "na"
        stime = getattr(rob, "solve_time", 0.0) if rob is not None else 0.0
        infeasible += int(status not in ["optimal", "optimal_inaccurate", "na"])
        solve_times.append(stime or 0.0)
        rows.append({"t": int(t), "x": x_next, "u": u, "rho": rho, "z": z, "xi": xi, "Tamb": tamb, "cost": cost, "solver_status": status})
    df = pd.DataFrame(rows)
    mets = compute_episode_metrics(df, safety_threshold=safety_threshold)
    mets.update({"infeasible_count": infeasible, "mean_solve_time": float(sum(solve_times)/len(solve_times))})
    return df, mets

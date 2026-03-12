from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cvxpy as cp


@dataclass
class SurrogateModel:
    a_h: float = 0.93
    b_h: float = 2.7
    c_h: float = 0.2
    d_h: float = 0.1
    a_c: float = 0.94
    b_c: float = 0.55
    d_c: float = 0.18


class BaseMainController:
    name = "base"

    def reset(self):
        pass

    def observe(self, state: dict, context: float, total_power: float):
        pass


class PIDController(BaseMainController):
    name = "pid"

    def __init__(self, target_hotspot=75.0, u_min=0.0, u_max=1.0, du_max=0.04, kp=0.05, ki=0.002, kd=0.015):
        self.target = target_hotspot
        self.u_min, self.u_max, self.du_max = u_min, u_max, du_max
        self.kp, self.ki, self.kd = kp, ki, kd
        self.reset()

    def reset(self):
        self.e_int = 0.0
        self.e_prev = 0.0
        self.u_prev = 0.3
        self.step_count = 0

    def act(self, state: dict, context: float, forecast_power: np.ndarray, Tamb_fc: np.ndarray, Tin_fc: np.ndarray):
        e = state["hotspot"] - self.target
        e_dot = e - self.e_prev
        cand_int = self.e_int + e
        raw = self.u_prev + self.kp * e + self.ki * cand_int + self.kd * e_dot
        sat = float(np.clip(raw, self.u_min, self.u_max))
        if abs(raw - sat) < 1e-8:
            self.e_int = cand_int
        du = float(np.clip(sat - self.u_prev, -self.du_max, self.du_max))
        u = float(np.clip(self.u_prev + du, self.u_min, self.u_max))
        self.e_prev, self.u_prev = e, u
        return u, 0.0, None


class PredictiveController(BaseMainController):
    name = "predictive"

    def __init__(self, surrogate: SurrogateModel, horizon=16, target_hotspot=75.0, u_min=0.0, u_max=1.0, du_max=0.04, w_u=0.8, w_du=10.0, w_slack=300.0, rho_fixed=0.0, replanning_interval=3):
        self.m = surrogate
        self.h = horizon
        self.target = target_hotspot
        self.u_min, self.u_max, self.du_max = u_min, u_max, du_max
        self.w_u, self.w_du, self.w_slack = w_u, w_du, w_slack
        self.rho_fixed = rho_fixed
        self.replanning_interval = replanning_interval
        self.reset()

    def reset(self):
        self.u_prev = 0.3
        self.step_count = 0

    def rho(self, context: float) -> float:
        return self.rho_fixed

    def _build_problem(self, H: int, h0: float, c0: float, p_fc: np.ndarray, tin_fc: np.ndarray, rho: float):
        u = cp.Variable(H)
        h = cp.Variable(H + 1)
        c = cp.Variable(H + 1)
        s = cp.Variable(H, nonneg=True)
        constr = [h[0] == h0, c[0] == c0]
        obj = 0
        up = self.u_prev
        for k in range(H):
            p_worst = float(p_fc[k] + rho)
            constr += [
                h[k + 1] == self.m.a_h * h[k] + self.m.b_h * p_worst - self.m.c_h * u[k] - self.m.d_h * c[k],
                c[k + 1] == self.m.a_c * c[k] + self.m.b_c * u[k] + self.m.d_c * float(tin_fc[k]),
                s[k] >= h[k + 1] - self.target,
                self.u_min <= u[k], u[k] <= self.u_max,
                -self.du_max <= u[k] - up, u[k] - up <= self.du_max,
            ]
            obj += self.w_u * cp.square(u[k]) + self.w_du * cp.square(u[k] - up) + self.w_slack * cp.square(s[k])
            up = u[k]
        return cp.Problem(cp.Minimize(obj), constr), u

    def _solve(self, h0: float, c0: float, p_fc: np.ndarray, tin_fc: np.ndarray, rho: float):
        H = min(self.h, len(p_fc))
        if H <= 0:
            return float(np.clip(self.u_prev, self.u_min, self.u_max)), {"status": "empty_horizon", "solve_time": 0.0}

        solver_attempts = [
            (cp.OSQP, {"warm_start": True, "verbose": False, "max_iter": 4000}),
            (cp.SCS, {"verbose": False, "max_iters": 3000, "eps": 1e-4}),
        ]

        last_status = "failed"
        total_solve_time = 0.0
        for solver, opts in solver_attempts:
            try:
                prob, u_var = self._build_problem(H, h0, c0, p_fc, tin_fc, rho)
                prob.solve(solver=solver, **opts)
                status = prob.status or "unknown"
                total_solve_time += float(prob.solver_stats.solve_time or 0.0)
                last_status = status
                if status in ["optimal", "optimal_inaccurate"] and u_var.value is not None:
                    u0 = float(np.clip(u_var.value[0], self.u_min, self.u_max))
                    return u0, {"status": status, "solve_time": total_solve_time, "solver": str(solver)}
            except Exception:
                continue

        return float(np.clip(self.u_prev, self.u_min, self.u_max)), {"status": f"fallback_prev_u:{last_status}", "solve_time": total_solve_time, "solver": "none"}

    def act(self, state: dict, context: float, forecast_power: np.ndarray, Tamb_fc: np.ndarray, Tin_fc: np.ndarray):
        rho = self.rho(context)
        if self.step_count % self.replanning_interval == 0:
            u, info = self._solve(state["hotspot"], state["t_coolant"], forecast_power, Tin_fc, rho)
            self.u_prev = u
            self.last_info = info
        else:
            u = self.u_prev
            info = self.last_info if hasattr(self, "last_info") else None
        self.step_count += 1
        return u, rho, info


class DeterministicMPCController(PredictiveController):
    name = "deterministic_mpc"


class RobustMPCController(PredictiveController):
    name = "robust_mpc"

    def __init__(self, *args, rho_uncertainty=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho_uncertainty = rho_uncertainty

    def rho(self, context: float) -> float:
        return self.rho_uncertainty


class NonContextualDROController(PredictiveController):
    name = "non_contextual_dro"

    def __init__(self, *args, rho_dro=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho_dro = rho_dro

    def rho(self, context: float) -> float:
        return self.rho_dro


class ContextualDROController(PredictiveController):
    name = "contextual_dro"

    def __init__(self, *args, rho_min=0.3, rho_max=2.2, rho_gain=1.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho_min, self.rho_max, self.rho_gain = rho_min, rho_max, rho_gain

    def rho(self, context: float) -> float:
        return float(np.clip(self.rho_min + self.rho_gain * max(0.0, context), self.rho_min, self.rho_max))


def tune_pid_on_validation(controller: PIDController, val_rollouts: list[dict], env_factory, forecast_steps: int = 12):
    candidates = [
        (0.04, 0.001, 0.01),
        (0.05, 0.002, 0.015),
        (0.06, 0.003, 0.018),
        (0.07, 0.004, 0.02),
    ]
    best = None
    for kp, ki, kd in candidates:
        controller.kp, controller.ki, controller.kd = kp, ki, kd
        score = 0.0
        for sc in val_rollouts[:6]:
            env = env_factory()
            controller.reset()
            env.reset()
            for t in sc["t"]:
                st = env.get_state()
                p_fc = np.full(forecast_steps, sc["q_map"][t].sum())
                tin_fc = np.full(forecast_steps, sc["Tin"][t])
                tamb_fc = np.full(forecast_steps, sc["Tamb"][t])
                u, _, _ = controller.act(st, float(sc["context"][t]), p_fc, tamb_fc, tin_fc)
                ns = env.step(u, float(sc["Tamb"][t]), float(sc["Tin"][t]), sc["q_map"][t])
                score += max(0.0, ns["hotspot"] - controller.target) * 20.0 + u * 2.0
        if best is None or score < best[0]:
            best = (score, kp, ki, kd)
    if best is not None:
        _, controller.kp, controller.ki, controller.kd = best
    return {"kp": controller.kp, "ki": controller.ki, "kd": controller.kd}

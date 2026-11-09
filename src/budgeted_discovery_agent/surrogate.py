"""A small Gaussian-process surrogate fitted on observed data only.

Exact GP regression with an RBF kernel on the simplex coordinates, a constant
mean equal to the observed mean, and a fixed noise level per fidelity. Screen
and confirm observations are treated as noisy views of one latent function,
which is exactly the assumption a deceptive screen bias breaks.
"""

from __future__ import annotations

import numpy as np

from .schemas import Candidate, Observation

DEFAULT_LENGTHSCALE = 0.25
NOISE_STD = {"screen": 0.10, "confirm": 0.05}


def _coords(cands: list[Candidate]) -> np.ndarray:
    return np.array([[c.a, c.b, c.c] for c in cands])


def farthest_point_rank(X: np.ndarray) -> list[int]:
    """Rank of each point in a farthest-point ordering that starts at the centroid.

    Used to break ties in ``rank``: with no observations every candidate has
    the same score, and ties then resolve to a space-filling set instead of
    neighbouring ids.
    """
    order = [int(np.argmin(((X - X.mean(0)) ** 2).sum(1)))]
    while len(order) < len(X):
        d = np.min(((X[:, None, :] - X[order][None, :, :]) ** 2).sum(-1), axis=1)
        d[order] = -1
        order.append(int(np.argmax(d)))
    rank = [0] * len(X)
    for r, i in enumerate(order):
        rank[i] = r
    return rank


class GPSurrogate:
    """Posterior over every candidate given the successful observations.

    Args:
        candidates: The full manifest.
        lengthscale: RBF lengthscale in simplex units.
    """

    def __init__(self, candidates: list[Candidate], lengthscale: float = DEFAULT_LENGTHSCALE):
        self.candidates = candidates
        self.ids = [c.candidate_id for c in candidates]
        self.lengthscale = lengthscale
        self._X = _coords(candidates)
        self._pos = {cid: i for i, cid in enumerate(self.ids)}
        self._spread = farthest_point_rank(self._X)

    def _k(self, a: np.ndarray, b: np.ndarray, var: float) -> np.ndarray:
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return var * np.exp(-d2 / (2 * self.lengthscale**2))

    def fit(self, observations: list[Observation]) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation for every candidate."""
        obs = [o for o in observations if o.status == "ok" and o.value is not None]
        n_all = len(self.ids)
        if not obs:
            return np.zeros(n_all), np.full(n_all, 0.3)
        idx = np.array([self._pos[o.candidate_id] for o in obs])
        y = np.array([o.value for o in obs], dtype=float)
        noise = np.array([NOISE_STD[o.fidelity] ** 2 for o in obs])
        mu0 = float(y.mean())
        var = max(float(y.var()), 0.01)
        Xo = self._X[idx]
        K = self._k(Xo, Xo, var) + np.diag(noise) + 1e-9 * np.eye(len(obs))
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - mu0))
        Ks = self._k(self._X, Xo, var)
        mean = mu0 + Ks @ alpha
        v = np.linalg.solve(L, Ks.T)
        sd = np.sqrt(np.clip(var - (v**2).sum(0), 1e-12, None))
        return mean, sd

    def calibration(self, observations: list[Observation]) -> dict[str, float | int] | None:
        """Leave-one-out diagnostics: mean |z| and one-sigma coverage."""
        obs = [o for o in observations if o.status == "ok" and o.value is not None]
        if len(obs) < 4:
            return None
        zs = []
        for i, o in enumerate(obs):
            rest = obs[:i] + obs[i + 1 :]
            mean, sd = self.fit(rest)
            p = self._pos[o.candidate_id]
            total = np.sqrt(sd[p] ** 2 + NOISE_STD[o.fidelity] ** 2)
            zs.append((o.value - mean[p]) / total)
        z = np.abs(np.array(zs))
        return {
            "loo_points": len(obs),
            "loo_mean_abs_z": round(float(z.mean()), 3),
            "loo_one_sigma_coverage": round(float((z <= 1).mean()), 3),
        }

    def rank(
        self,
        observations: list[Observation],
        strategy: str,
        n: int,
        exclude: set[str],
        beta: float = 1.0,
    ) -> list[tuple[str, float, float]]:
        """Top ``n`` candidates for a strategy: explore, exploit or balanced."""
        mean, sd = self.fit(observations)
        if strategy == "explore":
            score = sd
        elif strategy == "exploit":
            score = mean
        elif strategy == "balanced":
            score = mean + beta * sd
        else:
            raise ValueError(f"unknown strategy {strategy}")
        order = sorted(range(len(self.ids)), key=lambda i: (-round(score[i], 9), self._spread[i]))
        out = []
        for i in order:
            if self.ids[i] in exclude:
                continue
            out.append((self.ids[i], float(mean[i]), float(sd[i])))
            if len(out) == n:
                break
        return out

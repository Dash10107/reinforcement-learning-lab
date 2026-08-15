"""
Gaussian HMM fitting, BIC model selection, and regime inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["log_ret", "volatility", "momentum_5d", "rsi"]


def fit_hmm(
    feat: pd.DataFrame,
    n_states: int,
    cov_type: str = "diag",
    n_iter: int = 2000,
    seed: int = 42,
) -> tuple[GaussianHMM, StandardScaler, np.ndarray]:
    """Fit a Gaussian HMM and return (model, scaler, regime_sequence)."""
    X = feat[FEATURE_COLS].values.astype(np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=cov_type,
        n_iter=n_iter,
        tol=1e-5,
        random_state=seed,
    )
    model.fit(Xs)
    regimes = model.predict(Xs)
    return model, scaler, regimes


def bic_score(model: GaussianHMM, Xs: np.ndarray, n_states: int) -> float:
    """BIC = -2·log_likelihood + k·log(n)  (lower is better)."""
    n, d = Xs.shape
    # Free parameters: n*(n-1) transition, n*(n-1) start, n*d means, n*d variances (diag)
    k = n_states * (n_states - 1) + (n_states - 1) + n_states * d * 2
    log_lik = model.score(Xs) * n  # score() returns per-sample average
    return -2 * log_lik + k * np.log(n)


def select_n_regimes(
    feat: pd.DataFrame,
    min_n: int = 2,
    max_n: int = 5,
    seed: int = 42,
) -> tuple[int, list[tuple[int, float]]]:
    """Use BIC to select optimal number of regimes."""
    X = feat[FEATURE_COLS].values.astype(np.float64)
    Xs = StandardScaler().fit_transform(X)

    scores = []
    for n in range(min_n, max_n + 1):
        try:
            m = GaussianHMM(
                n_components=n, covariance_type="diag", n_iter=1000, random_state=seed
            )
            m.fit(Xs)
            bic = bic_score(m, Xs, n)
            scores.append((n, bic))
        except Exception:
            pass

    if not scores:
        return 3, []

    best_n = min(scores, key=lambda x: x[1])[0]
    return best_n, scores


def current_regime_probs(
    model: GaussianHMM,
    scaler: StandardScaler,
    feat: pd.DataFrame,
) -> np.ndarray:
    """Return posterior probability distribution over regimes for last observation."""
    X = feat[FEATURE_COLS].values.astype(np.float64)
    Xs = scaler.transform(X)
    # Use forward algorithm log-alpha
    log_proba = model.predict_proba(Xs)
    return log_proba[-1]  # last time step


def model_diagnostics(
    model: GaussianHMM,
    scaler: StandardScaler,
    feat: pd.DataFrame,
    n_states: int,
) -> dict:
    """Compute model quality metrics."""
    X = feat[FEATURE_COLS].values.astype(np.float64)
    Xs = scaler.transform(X)
    n = len(Xs)

    log_lik = model.score(Xs) * n
    k = n_states * (n_states - 1) + (n_states - 1) + n_states * len(FEATURE_COLS) * 2
    aic = -2 * log_lik + 2 * k
    bic = -2 * log_lik + k * np.log(n)

    return {
        "log_likelihood": round(log_lik, 2),
        "aic": round(aic, 2),
        "bic": round(bic, 2),
        "n_observations": n,
        "n_features": len(FEATURE_COLS),
        "n_states": n_states,
        "converged": model.monitor_.converged,
    }

"""ReadPrune fixed-protocol CPU set mathematics.

No decoder, selector-private score, labels, training, or token modification.
All geometry and set arithmetic is float64. Reference checks use fresh
Cholesky factorizations of the *same* fixed-ridge Gram matrix; no ridge
inflation or Schur-complement clipping is permitted.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular

ETA = 1e-6
DEADZONE = 1e-6
# Numerical adjudication only, not quality hyperparameters.
REFERENCE_GUARD = 1e-7
TIE_ATOL = 1e-10
FORMULA_VERSION = "ReadPrune_FP64_exactK_v1"


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu", dtype=__import__("torch").float64).numpy()
    return np.asarray(value, dtype=np.float64)


def quality(delta):
    delta = _array(delta)
    if delta.ndim != 1 or not np.isfinite(delta).all() or len(delta) == 0:
        raise ValueError("delta must be a nonempty finite vector")
    s = np.maximum(np.abs(delta) - DEADZONE, 0.0)
    mu = float(np.mean(s))
    if not math.isfinite(mu):
        raise FloatingPointError("nonfinite mean sensitivity")
    q = np.ones_like(s) if mu == 0.0 else 1.0 + s / (s + mu)
    if not np.isfinite(q).all() or np.any(q < 1.0) or np.any(q >= 2.0):
        raise FloatingPointError("quality outside [1, 2)")
    return q, {
        "positive_count": int(np.count_nonzero(delta > 0.0)),
        "negative_count": int(np.count_nonzero(delta < 0.0)),
        "zero_count": int(np.count_nonzero(delta == 0.0)),
        "deadzone_count": int(np.count_nonzero(np.abs(delta) <= DEADZONE)),
        "mu": mu,
        "flat": bool(mu == 0.0),
        "s_min": float(s.min()), "s_max": float(s.max()),
        "q_min": float(q.min()), "q_max": float(q.max()),
        "deadzone": DEADZONE,
    }


def geometry(X):
    X = _array(X)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a nonempty token-by-feature matrix")
    if not np.isfinite(X).all():
        raise ValueError("nonfinite raw projector features")
    norms = np.linalg.norm(X, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError("zero or invalid raw projector token norm")
    U = X / norms[:, None]
    B = U @ U.T
    B.flat[::B.shape[0] + 1] += ETA
    if not np.isfinite(B).all():
        raise FloatingPointError("nonfinite FP64 Gram matrix")
    return np.asarray(B, dtype=np.float64, order="C")


def _validate(B, q):
    B, q = _array(B), _array(q)
    if B.ndim != 2 or B.shape[0] != B.shape[1] or B.shape[0] == 0:
        raise ValueError("B must be a nonempty square matrix")
    if q.shape != (len(B),):
        raise ValueError("q must have one entry per token")
    if not np.isfinite(B).all() or not np.isfinite(q).all() or np.any(q <= 0):
        raise ValueError("B and positive q must be finite")
    if not np.allclose(B, B.T, atol=1e-12, rtol=0.0):
        raise ValueError("asymmetric Gram matrix")
    if np.any(np.diag(B) <= 0.0):
        raise ValueError("nonpositive Gram diagonal")
    return B, q


def _support(S, n):
    raw = list(S)
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in raw):
        raise ValueError("support indices must be integers")
    S = np.asarray(raw, dtype=np.int64)
    if len(set(raw)) != len(raw) or np.any(S < 0) or np.any(S >= n):
        raise ValueError("support indices must be unique and in range")
    return S


def _objective(B, logq, S):
    if len(S) == 0:
        return 0.0
    L = cholesky(B[np.ix_(S, S)], lower=True, check_finite=False)
    result = 2.0 * float(logq[S].sum()) + 2.0 * float(np.log(np.diag(L)).sum())
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite direct objective")
    return result


def objective(B, q, S):
    B, q = _validate(B, q)
    S = _support(S, len(B))
    return _objective(B, np.log(q), S)


def _reference_choice(candidate_values):
    """Values have already been directly refactored; raster lexicographic ties."""
    highest = max(row[0] for row in candidate_values)
    tied = [row for row in candidate_values if highest - row[0] <= TIE_ATOL]
    return min(tied, key=lambda row: row[1])


def select(B, q, k=64):
    B, q = _validate(B, q)
    n = len(B)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 0 < k <= n:
        raise ValueError("k must be an integer in [1, N]")
    logq = np.log(q)
    coeff = np.zeros((int(k), n), dtype=np.float64)
    residual = np.diag(B).copy()
    chosen = np.zeros(n, dtype=bool)
    order, trace = [], []
    current_F = 0.0
    reference_calls = 0
    max_reference_error = 0.0
    for step in range(int(k)):
        remaining = np.flatnonzero(~chosen)
        if not np.isfinite(residual[remaining]).all() or np.any(residual[remaining] <= 0):
            raise FloatingPointError("invalid greedy Schur complement; no clipping allowed")
        gains = 2.0 * logq[remaining] + np.log(residual[remaining])
        top = float(gains.max())
        contenders = remaining[gains >= top - REFERENCE_GUARD]
        reference_values = []
        for i in contenders:
            trial = np.asarray(order + [int(i)], dtype=np.int64)
            trial_F = _objective(B, logq, trial)
            reference_calls += 1
            direct_gain = trial_F - current_F
            analytic_gain = 2.0 * float(logq[i]) + math.log(float(residual[i]))
            error = abs(direct_gain - analytic_gain)
            max_reference_error = max(max_reference_error, error)
            if error > REFERENCE_GUARD:
                raise FloatingPointError(
                    f"greedy analytic/reference disagreement {error:.3g}; no silent repair"
                )
            reference_values.append((direct_gain, (int(i),), trial_F, analytic_gain))
        direct_gain, identity, next_F, analytic_gain = _reference_choice(reference_values)
        i = identity[0]
        denom = math.sqrt(float(residual[i]))
        cross = B[i].copy()
        if step:
            cross -= coeff[:step, i] @ coeff[:step]
        coeff[step] = cross / denom
        residual -= coeff[step] * coeff[step]
        chosen[i] = True
        order.append(i)
        trace.append({
            "step": step + 1, "add": i,
            "gain": float(direct_gain), "analytic_gain": float(analytic_gain),
            "objective_before": float(current_F), "objective_after": float(next_F),
            "reference_contenders": int(len(contenders)),
        })
        current_F = float(next_F)
    support = sorted(order)
    final_F = _objective(B, logq, np.asarray(support, dtype=np.int64))
    if abs(final_F - current_F) > REFERENCE_GUARD:
        raise FloatingPointError("sorted-support objective disagreement")
    return {
        "support": support, "order": order, "trace": trace,
        "objective": float(final_F),
        "reference_calls": int(reference_calls),
        "max_reference_error": float(max_reference_error),
        "numeric": {"dtype": "float64", "eta": ETA,
                    "reference_guard": REFERENCE_GUARD, "tie_atol": TIE_ATOL},
    }

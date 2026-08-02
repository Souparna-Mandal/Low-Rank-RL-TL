"""Order-r linear recurrence (AR) fitting and prediction for value sequences.

If a sequence's Hankel matrix has rank <= r it obeys q_t = sum_j c_j q_{t-j}
(Kronecker); these helpers fit the c_j by least squares and extrapolate with
them, which is the direct test of that property's predictive use.
"""
import numpy as np


def fit_ar(seqs, order, ridge=1e-8, intercept=False):
    """Least-squares fit of q_t = sum_{j=1..order} c_j q_{t-j} (+ c_0) over all
    valid windows of the given sequences.

    seqs: iterable of 1-D arrays. Returns coeffs of shape (order,) with
    coeffs[0] the weight on the most recent value, or (order+1,) with the
    intercept last when intercept=True.
    """
    X, y = [], []
    for s in seqs:
        s = np.asarray(s, dtype=np.float64)
        for t in range(order, len(s)):
            X.append(s[t - order:t][::-1])
            y.append(s[t])
    if not X:
        raise ValueError("no window of length order+1 in seqs")
    X, y = np.asarray(X), np.asarray(y)
    if intercept:
        X = np.hstack([X, np.ones((len(X), 1))])
    A = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def predict_one_step(coeffs, seq, intercept=False):
    """One-step-ahead predictions from the *true* history: returns array
    aligned with seq[order:]."""
    c, c0 = (coeffs[:-1], coeffs[-1]) if intercept else (coeffs, 0.0)
    r = len(c)
    s = np.asarray(seq, dtype=np.float64)
    preds = [c @ s[t - r:t][::-1] + c0 for t in range(r, len(s))]
    return np.asarray(preds)


def free_run(coeffs, seed_vals, horizon, intercept=False):
    """Recursive multi-step prediction: extrapolate `horizon` values from the
    last r seed values, feeding predictions back in."""
    c, c0 = (coeffs[:-1], coeffs[-1]) if intercept else (coeffs, 0.0)
    r = len(c)
    hist = list(np.asarray(seed_vals, dtype=np.float64)[-r:])
    if len(hist) < r:
        raise ValueError("need at least `order` seed values")
    out = []
    for _ in range(horizon):
        nxt = c @ np.asarray(hist[-r:][::-1]) + c0
        out.append(nxt)
        hist.append(nxt)
    return np.asarray(out)


def nrmse(pred, true):
    """RMSE normalised by the RMS of the true signal (scale-free)."""
    pred, true = np.asarray(pred), np.asarray(true)
    rms = np.sqrt(np.mean(true ** 2))
    return float(np.sqrt(np.mean((pred - true) ** 2)) / max(rms, 1e-12))

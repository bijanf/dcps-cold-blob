"""Multiple-testing correction utilities.

Benjamini--Hochberg FDR (Benjamini & Hochberg 1995).  No external
dependency on statsmodels so this stays self-contained in dcps.
"""

from __future__ import annotations

import numpy as np


def fdr_bh(pvals, alpha: float = 0.05) -> tuple:
    """Benjamini--Hochberg step-up FDR.

    Parameters
    ----------
    pvals : 1-D array-like of raw p-values.  NaNs are ignored and
            returned as NaN q-values.
    alpha : nominal FDR level (default 0.05).

    Returns
    -------
    rejected : bool ndarray; True where the BH-adjusted q-value is
               <= alpha.
    q : ndarray of BH-adjusted q-values, same shape as input.
    """
    p = np.asarray(pvals, dtype=float)
    out_q = np.full_like(p, np.nan)
    out_rej = np.zeros_like(p, dtype=bool)
    finite = np.isfinite(p)
    if not finite.any():
        return out_rej, out_q
    pf = p[finite]
    m = pf.size
    order = np.argsort(pf)
    ranks = np.arange(1, m + 1)
    q_sorted = pf[order] * m / ranks
    # Enforce monotonicity (running min from the right).
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_out = np.empty(m)
    q_out[order] = q_sorted
    q_out = np.clip(q_out, 0.0, 1.0)
    out_q[finite] = q_out
    out_rej[finite] = q_out <= alpha
    return out_rej, out_q


__all__ = ["fdr_bh"]

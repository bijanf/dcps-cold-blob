"""Kraskov-Stoegbauer-Grassberger (KSG) transfer entropy estimator.

Implements the Frenzel & Pompe (2007) extension of the KSG mutual-information
estimator (Kraskov, Stoegbauer & Grassberger, 2004) to conditional MI, which
gives directly the transfer entropy

    T_{Y -> X} = I(X_{t+1}; Y_t^{(l)} | X_t^{(k)})

with history lengths k and l.  We follow the implementation conventions of
Wibral et al. (2014) and the IDTxl package: max-norm metric, k-th nearest
neighbour in the joint space, marginal counts via tree range searches, and
the digamma correction

    T_{Y->X} = psi(k_NN) + < psi(n_z + 1) - psi(n_xz + 1) - psi(n_yz + 1) >

where the marginal counts (n_z, n_xz, n_yz) for each query point are the
numbers of points strictly inside the max-norm ball whose radius equals the
joint-space distance to the k-th nearest neighbour.  In the Frenzel-Pompe
convention, x = X_{t+1} (target's next state), y = Y_t (source history),
z = X_t (target history).

References:
    Kraskov A., Stoegbauer H., Grassberger P. (2004).
    Frenzel S., Pompe B. (2007). Phys. Rev. Lett. 99, 204101.
    Wibral M. et al. (2014). PLoS ONE 9, e102833.

Notes:
    * Inputs are standardised to unit variance before tree construction;
      this matches NPEET / IDTxl defaults and improves numerical stability.
    * A small uniform jitter (1e-10) is added to break degeneracies, again
      matching standard practice.
    * The estimator is asymptotically unbiased; for short series it can
      be slightly biased and is best used with surrogate testing.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


def _embed(x: np.ndarray, hist: int) -> np.ndarray:
    """Build delay-embedded history matrix.  Returns (T-hist, hist)."""
    n = x.shape[0]
    out = np.empty((n - hist, hist), dtype=np.float64)
    for i in range(hist):
        out[:, i] = x[hist - 1 - i: n - 1 - i]
    return out


def ksg_te(
    source: np.ndarray,
    target: np.ndarray,
    k: int = 1,
    ell: int = 1,
    k_nn: int = 4,
    seed: int = 0,
    standardize: bool = True,
) -> float:
    """Estimate transfer entropy T_{Y -> X} using KSG / Frenzel-Pompe CMI.

    Parameters
    ----------
    source : (T,) array     Y, candidate driver.
    target : (T,) array     X, candidate response.
    k, ell : int            History lengths for X and Y. Defaults k=l=1.
    k_nn   : int            Nearest-neighbour parameter (KSG hyperparameter).
                            4 is the standard default.
    seed   : int            RNG seed for jitter.
    standardize : bool      Whether to z-score inputs before estimation.

    Returns
    -------
    Transfer entropy in nats.  (Multiply by 1/ln(2) to get bits.)
    """
    src = np.asarray(source, dtype=np.float64).ravel()
    tgt = np.asarray(target, dtype=np.float64).ravel()
    if src.shape[0] != tgt.shape[0]:
        raise ValueError("source and target length mismatch")
    if not (np.isfinite(src).all() and np.isfinite(tgt).all()):
        return float("nan")

    if standardize:
        s_src = src.std()
        s_tgt = tgt.std()
        if s_src == 0 or s_tgt == 0:
            return float("nan")
        src = (src - src.mean()) / s_src
        tgt = (tgt - tgt.mean()) / s_tgt

    rng = np.random.default_rng(seed)
    src = src + 1e-10 * rng.standard_normal(src.shape)
    tgt = tgt + 1e-10 * rng.standard_normal(tgt.shape)

    hist = max(k, ell)
    Xnext = tgt[hist:].reshape(-1, 1)                     # (N, 1)  X_{t+1}
    Xhist = _embed(tgt, hist)[:, :k]                       # (N, k)  X_t^{(k)}
    Yhist = _embed(src, hist)[:, :ell]                     # (N, l)  Y_t^{(l)}

    N = Xnext.shape[0]
    if N <= k_nn + 2:
        return float("nan")

    Z = Xhist                          # condition variable
    XZ = np.hstack([Xnext, Xhist])
    YZ = np.hstack([Yhist, Xhist])
    XYZ = np.hstack([Xnext, Yhist, Xhist])

    tree_xyz = cKDTree(XYZ)
    # Joint-space k-NN distances using max-norm (Chebyshev = p=inf)
    eps, _ = tree_xyz.query(XYZ, k=k_nn + 1, p=np.inf)
    eps = eps[:, -1]

    # For each marginal, count points strictly within eps under max-norm,
    # excluding the query itself (KDTree.query_ball_point returns the
    # query, so we subtract 1 below).
    tree_z = cKDTree(Z)
    tree_xz = cKDTree(XZ)
    tree_yz = cKDTree(YZ)

    n_z = np.array([len(tree_z.query_ball_point(Z[i], eps[i] - 1e-12, p=np.inf)) - 1
                    for i in range(N)])
    n_xz = np.array([len(tree_xz.query_ball_point(XZ[i], eps[i] - 1e-12, p=np.inf)) - 1
                     for i in range(N)])
    n_yz = np.array([len(tree_yz.query_ball_point(YZ[i], eps[i] - 1e-12, p=np.inf)) - 1
                     for i in range(N)])

    # Frenzel-Pompe formula for conditional MI in nats
    te_nats = (digamma(k_nn)
               + np.mean(digamma(n_z + 1) - digamma(n_xz + 1) - digamma(n_yz + 1)))
    return float(te_nats)


def ksg_te_bits(source: np.ndarray, target: np.ndarray, **kwargs) -> float:
    """Return KSG TE in bits (more comparable to the binary-discretised
    estimator used elsewhere)."""
    return ksg_te(source, target, **kwargs) / np.log(2.0)

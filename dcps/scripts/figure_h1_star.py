"""Figure 4: H1* spatial result -- the static Quiescence Hypothesis confirmed.

Panel (a): scatter of <r_loc>_t vs <|grad SSH|>_t across NA cells, two products
Panel (b): maps of the two fields side by side for ORAS5
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Rectangle

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.order_parameter import local_r
from dcps.phase import analytic_signal, edge_trim


MULTI = CACHE_DIR / "multi"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"


def compute_fields(product: str) -> tuple[xr.DataArray, xr.DataArray]:
    p1 = xr.open_dataset(MULTI / f"phase1_{product}.nc")
    _, _, phi = analytic_signal(p1.sst_anom)
    phi = edge_trim(phi)
    rl = local_r(phi, radius_km=500.0).mean("time")
    ssh_mean = p1.ssh_raw.mean("time")
    grad = np.sqrt(ssh_mean.differentiate("lat") ** 2
                    + ssh_mean.differentiate("lon") ** 2)
    p1.close()
    return rl, grad


def main():
    fig = plt.figure(figsize=(12, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    ax_scatter = fig.add_subplot(gs[0, :])
    ax_rl = fig.add_subplot(gs[1, 0])
    ax_grad = fig.add_subplot(gs[1, 1])

    products = ["oras5", "glorys12"]
    colours = {"oras5": "C0", "glorys12": "C3"}

    for prod in products:
        rl, grad = compute_fields(prod)
        a = rl.values.ravel()
        b = grad.values.ravel()
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        ax_scatter.scatter(b, a, s=8, alpha=0.35, color=colours[prod], label=prod)
    ax_scatter.set_xlabel(r"$|\nabla \overline{\mathrm{SSH}}|$  (m / deg)")
    ax_scatter.set_ylabel(r"$\langle r_\mathrm{loc}\rangle_t$")
    ax_scatter.legend(loc="upper right", fontsize=9, frameon=False)
    ax_scatter.grid(alpha=0.25)

    rl_o, grad_o = compute_fields("oras5")

    im = ax_rl.pcolormesh(rl_o.lon, rl_o.lat, rl_o, cmap="RdYlBu_r",
                          vmin=0, vmax=1, shading="auto")
    plt.colorbar(im, ax=ax_rl, label=r"$\langle r_\mathrm{loc}\rangle_t$")
    ax_rl.set_xlabel("longitude")
    ax_rl.set_ylabel("latitude")
    ax_rl.add_patch(Rectangle((-40, 45), 25, 15, fill=False, edgecolor="k", lw=1.2))
    ax_rl.add_patch(Rectangle((-60, 20), 50, 20, fill=False, edgecolor="k",
                                lw=1.0, ls="--"))

    im = ax_grad.pcolormesh(grad_o.lon, grad_o.lat, grad_o, cmap="magma",
                             shading="auto")
    plt.colorbar(im, ax=ax_grad, label=r"$|\nabla \overline{\mathrm{SSH}}|$ (m/deg)")
    ax_grad.set_xlabel("longitude")
    ax_grad.set_ylabel("latitude")
    ax_grad.add_patch(Rectangle((-40, 45), 25, 15, fill=False, edgecolor="w", lw=1.2))
    ax_grad.add_patch(Rectangle((-60, 20), 50, 20, fill=False, edgecolor="w",
                                  lw=1.0, ls="--"))

    out = MANUSCRIPT_FIGS / "fig4_quiescence_spatial.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

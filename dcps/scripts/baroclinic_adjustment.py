"""Baroclinic adjustment timescale figure.

Plan Step 6: solve the 1-D vertical diffusion equation
    dT/dt = kappa_v * d^2T/dz^2
on a semi-infinite column with a step surface boundary condition
T(z=0, t>=0) = T0 (representing an abrupt anthropogenic cooling
imposed on the upper ocean) and T(z, 0) = 0.  The analytic
solution is the complementary error function
    T(z, t) = T0 * erfc( z / (2 sqrt(kappa_v t)) ) .

Plot T(z, t) for t = 50, 100, 200 years, with two diffusivities:
  kappa_v = 1e-4 m^2/s   (upper-ocean thermocline)
  kappa_v = 1e-5 m^2/s   (deep ocean abyss)

Overlay the approximate depths of the seven testable Caesar-2021
multi-proxy records and the two augmented deep proxies, indicating
which proxies should and should not yet register the modern surface
signal.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

from dcps.config import CACHE_DIR, PKG_ROOT
from dcps.nature_style import apply_nature_style
apply_nature_style()

OUT_DIR = CACHE_DIR / "baroclinic"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"

# Proxies (depth in m, label, category) -- approximate depths used in
# the manuscript's depth-split discussion.
PROXIES = [
    (   0,  "OISST / ORAS5 surface",       "surface"),
    ( 100,  "Reynolds T_sub",               "thermocline"),
    ( 200,  "T. quinqueloba (Holland)",    "thermocline"),
    ( 800,  "Spooner 2020",                  "deep"),
    (1500,  "Lippold 2019",                  "deep"),
    (2400,  "Thornalley 2018 SS",            "deep"),
    (3484,  "Moffa-Sanchez 2015 RAPiD-35",   "deep"),
    (3700,  "Thibodeau 2018 MD99-2220",      "deep"),
]


def T_over_T0(z, t_yr, kappa_v):
    """Solution to dT/dt = kappa_v * d2T/dz2 with step BC.
    T(z,t)/T0 = erfc(z / (2 sqrt(kappa_v * t))).
    z in m, t in years, kappa_v in m^2/s.
    """
    t_s = np.asarray(t_yr) * 365.25 * 86400.0
    arg = z[:, None] / (2.0 * np.sqrt(kappa_v * t_s))
    return erfc(arg)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" Baroclinic adjustment via 1-D vertical diffusion")
    print("=" * 70)

    z = np.linspace(0, 4000, 401)  # m
    ts = [50, 100, 200]            # yr
    kappas = {
        "upper-ocean ($\\kappa_v = 10^{-4}$ m$^2$/s)": 1.0e-4,
        "deep ocean ($\\kappa_v = 10^{-5}$ m$^2$/s)": 1.0e-5,
    }

    # Compute T/T0 at the proxy depths for the deep-ocean diffusivity
    z_proxy = np.array([p[0] for p in PROXIES], dtype=float)
    T_deep_50 = erfc(z_proxy / (2 * np.sqrt(1e-5 * 50 * 365.25 * 86400)))
    T_deep_100 = erfc(z_proxy / (2 * np.sqrt(1e-5 * 100 * 365.25 * 86400)))
    T_deep_200 = erfc(z_proxy / (2 * np.sqrt(1e-5 * 200 * 365.25 * 86400)))
    T_upper_200 = erfc(z_proxy / (2 * np.sqrt(1e-4 * 200 * 365.25 * 86400)))

    print("\n  At t=200 yr, T/T0 at each proxy depth:")
    print(f"  {'depth':>6} {'kappa=1e-4':>11} {'kappa=1e-5':>11}  proxy")
    print("  " + "-" * 65)
    rows = []
    for (zp, lbl, cat), tu200, td50, td100, td200 in zip(
        PROXIES, T_upper_200, T_deep_50, T_deep_100, T_deep_200):
        print(f"  {zp:>5}m {tu200:>11.3f} {td200:>11.3f}  {lbl} ({cat})")
        rows.append(dict(depth_m=float(zp), label=lbl, category=cat,
                          T_upper_t200=float(tu200),
                          T_deep_t50=float(td50),
                          T_deep_t100=float(td100),
                          T_deep_t200=float(td200)))

    with open(OUT_DIR / "baroclinic.json", "w") as f:
        json.dump(dict(proxies=rows,
                       T0_definition="step surface T anomaly",
                       kappas=dict(upper=1e-4, deep=1e-5),
                       integration_times_yr=ts), f, indent=2)
    print(f"\nWrote {OUT_DIR / 'baroclinic.json'}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.0),
                              constrained_layout=True, sharey=True)
    colors = ["C0", "C1", "C2"]
    for ax, (label, kv) in zip(axes, kappas.items()):
        for t, c in zip(ts, colors):
            T = T_over_T0(z, t, kv).flatten()
            ax.plot(T, z, color=c, lw=2, label=f"$t = {t}$ yr")
        # Proxy markers
        for (zp, lbl, cat) in PROXIES:
            marker = {"surface": "o", "thermocline": "s",
                       "deep": "^"}[cat]
            c_proxy = {"surface": "k", "thermocline": "#d6604d",
                        "deep": "#4393c3"}[cat]
            ax.plot(1.05, zp, marker=marker, ms=7, color=c_proxy,
                     clip_on=False)
        ax.invert_yaxis()
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(4000, 0)
        ax.set_xlabel(r"$T(z, t) / T_{0}$ (fraction of surface anomaly)")
        ax.set_title(label, fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("depth (m)")

    # Annotate proxies on the right of the second panel
    ax = axes[1]
    for (zp, lbl, cat) in PROXIES:
        ax.annotate(lbl, xy=(1.05, zp), xytext=(1.10, zp),
                     fontsize=8, va="center",
                     annotation_clip=False)

    # Legend for proxy categories
    from matplotlib.lines import Line2D
    proxy_legend = [
        Line2D([0], [0], marker="o", color="w",
                markerfacecolor="k", markersize=7, label="surface"),
        Line2D([0], [0], marker="s", color="w",
                markerfacecolor="#d6604d", markersize=7,
                label="thermocline"),
        Line2D([0], [0], marker="^", color="w",
                markerfacecolor="#4393c3", markersize=7,
                label=r"deep ($\geq$ 800 m)"),
    ]
    axes[1].legend(handles=proxy_legend, loc="lower left",
                    fontsize=9, framealpha=0.9, title="proxies")

    fig.suptitle("1-D vertical-diffusion adjustment to a step surface "
                  "anomaly", fontsize=12)
    out_fig = MANUSCRIPT_FIGS / "fig_baroclinic_adjustment.pdf"
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()

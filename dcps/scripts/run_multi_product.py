"""Multi-reanalysis Phase 1+2+H1 pipeline.

For each available product (ORAS5, GLORYS12, ECCO), run Phase 1 (regrid +
preprocess), Phase 2 (Hilbert + Kuramoto R(t)), and the H1 correlation test
against RAPID. Persist per-product caches and emit a comparison table +
figure.

Per the user's domain-knowledge correction, all products are restricted to
the Argo era (>= 2000). Each product's actual native window is intersected
with this restriction.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from dcps import anomaly, regrid
from dcps.config import CACHE_DIR, GRID_DEG, PKG_ROOT
from dcps.h1 import lowpass_1yr, standardise, test_h1
from dcps.io import load_rapid_amoc
from dcps.order_parameter import global_R, global_R_pooled
from dcps.phase import analytic_signal, edge_trim
from dcps.products import PRODUCTS, load_product_var, product_window


MULTI_DIR = CACHE_DIR / "multi"
MANUSCRIPT_FIGS = PKG_ROOT.parent / "manuscript" / "figs"
ARGO_START = "2000-01-01"


def process_one_product(product: str) -> dict | None:
    """Run Phase 1+2 on one product, return its R(t) variants and cache path."""
    s, e = product_window(product)
    s = max(s, ARGO_START)
    print(f"\n{'='*60}\n{product.upper()}: window {s}..{e}\n{'='*60}")

    available = list(PRODUCTS[product]["vars"].keys())
    print(f"  available aliases: {available}")

    out = {"product": product, "start": s, "end": e}
    raw = {}
    anom = {}
    for alias in available:
        try:
            t0 = time.time()
            da = load_product_var(product, alias, start=s, end=e)
            n_t = da.sizes.get("time", da.sizes.get("time_counter", 0))
            print(f"  [{alias}] loaded {tuple(da.sizes.values())} in {time.time()-t0:.1f}s "
                  f"(n_t = {n_t})")
        except Exception as exc:
            print(f"  [{alias}] LOAD FAILED: {exc}")
            continue
        # rename time_counter -> time if needed (ORAS5 path actually already does this)
        if "time_counter" in da.dims:
            da = da.rename({"time_counter": "time"})
        t0 = time.time()
        rg = regrid.regrid_to_2deg(da, grid_deg=GRID_DEG).rename(f"{alias}_raw")
        print(f"  [{alias}] regrid {tuple(rg.sizes.values())} in {time.time()-t0:.1f}s")
        ana = anomaly.preprocess_pipeline(rg).rename(f"{alias}_anom")
        raw[alias] = rg
        anom[alias] = ana

    if not raw:
        print(f"  no usable data; skipping {product}.")
        return None

    # Phase 1 cache
    MULTI_DIR.mkdir(parents=True, exist_ok=True)
    p1 = MULTI_DIR / f"phase1_{product}.nc"
    ds_p1 = xr.Dataset({**{f"{a}_raw": raw[a] for a in raw}, **{f"{a}_anom": anom[a] for a in anom}})
    ds_p1.attrs["product"] = product
    ds_p1.attrs["window"] = f"{s} .. {e}"
    enc = {v: {"zlib": True, "complevel": 4} for v in ds_p1.data_vars}
    ds_p1.to_netcdf(p1, encoding=enc)

    # Phase 2: Hilbert + R(t)
    out["R_variants"] = {}
    phase_fields = {}
    for alias, ana in anom.items():
        _, _, phi = analytic_signal(ana)
        phi = edge_trim(phi)
        phase_fields[alias] = phi
        R = global_R(phi).rename(f"R_{alias}")
        out["R_variants"][f"R_{alias}"] = R

    if "sst" in phase_fields and "ssh" in phase_fields:
        a, b = phase_fields["sst"], phase_fields["ssh"]
        if a.sizes["time"] == b.sizes["time"]:
            out["R_variants"]["R_pooled"] = global_R_pooled(a, b).rename("R_pooled")

    # Save R cache
    p2 = MULTI_DIR / f"phase2_R_{product}.nc"
    R_ds = xr.merge(list(out["R_variants"].values()), compat="override")
    R_ds.attrs["product"] = product
    R_ds.to_netcdf(p2, encoding={v: {"zlib": True} for v in R_ds.data_vars})
    print(f"  R(t) variants: " + ", ".join(
        f"{k}={float(v.mean()):.3f}" for k, v in out["R_variants"].items()
    ))

    out["phase1_cache"] = str(p1)
    out["phase2_cache"] = str(p2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--products", default="oras5,glorys12,ecco",
                   help="Comma-separated product list")
    args = p.parse_args()
    requested = [p.strip() for p in args.products.split(",") if p.strip()]

    results = {}
    for prod in requested:
        if prod not in PRODUCTS:
            print(f"Unknown product {prod!r}; skip.")
            continue
        try:
            r = process_one_product(prod)
            if r:
                results[prod] = r
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"\n{prod}: FAILED — {exc}")

    # H1 across products
    rapid = load_rapid_amoc()
    h1_table = []
    print(f"\n{'='*60}\nH1 multi-product comparison (vs RAPID 26.5N)\n{'='*60}")
    print(f"{'product':<10} {'variant':<12} {'rho':>6} {'95% CI':>20} {'p':>9}  {'n':>5}  verdict")
    print("-" * 80)
    for prod, r in results.items():
        for vname, R_da in r["R_variants"].items():
            try:
                res = test_h1(R_da, rapid)
                verdict = ("strong" if res.pass_strong else
                           "suggestive" if res.pass_suggestive else "falsified")
                row = (prod, vname, res.rho, res.bootstrap_low, res.bootstrap_high,
                       res.p_value, res.n_months, verdict)
                h1_table.append(row)
                print(f"{prod:<10} {vname:<12} {res.rho:+.3f} "
                      f" ({res.bootstrap_low:+.3f}, {res.bootstrap_high:+.3f}) "
                      f"{res.p_value:.2e}  {res.n_months:>4}  {verdict}")
            except Exception as exc:
                print(f"{prod:<10} {vname:<12}  -- error: {exc}")

    # Persist results JSON
    out_json = MULTI_DIR / "h1_multi_product.json"
    with open(out_json, "w") as f:
        json.dump([{"product": r[0], "variant": r[1], "rho": r[2],
                    "ci_low": r[3], "ci_high": r[4], "p": r[5],
                    "n": r[6], "verdict": r[7]} for r in h1_table],
                  f, indent=2)
    print(f"\nWrote {out_json}")

    # ===== Comparison figure =====
    if not h1_table:
        return
    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                              gridspec_kw={"width_ratios": [1.6, 1.0]},
                              constrained_layout=True)

    # (a) Forest plot: rho with 95% CI per (product, variant)
    a = axes[0]
    labels = [f"{r[0]}/{r[1]}" for r in h1_table]
    rhos = [r[2] for r in h1_table]
    los = [r[3] for r in h1_table]
    his = [r[4] for r in h1_table]
    ys = np.arange(len(labels))
    a.errorbar(rhos, ys, xerr=[np.array(rhos) - np.array(los), np.array(his) - np.array(rhos)],
               fmt="o", color="C0", capsize=3)
    a.axvline(0, color="grey", lw=0.5)
    a.axvline(0.6, color="C2", ls="--", lw=0.8, label=r"H1 threshold $\rho^* = 0.6$")
    a.axvline(-0.6, color="C2", ls="--", lw=0.8)
    a.set_yticks(ys); a.set_yticklabels(labels)
    a.set_xlabel(r"Pearson $\rho$ (R(t) vs RAPID 26.5$^\circ$N AMOC)")
    a.set_xlim(-0.7, 0.9)
    a.set_title("(a) H1 multi-product forest plot (95% block-bootstrap CI)")
    a.legend(loc="upper right", fontsize=8)
    a.invert_yaxis()

    # (b) Time series overlay for the strongest-correlated variant per product
    a = axes[1]
    common = None
    overlap_start = max(r[0] for r in [(rapid.time.min().values,)] + [(xr.open_dataset(r["phase2_cache"]).time.min().values,) for _, r in results.items()])
    # Just plot R_pooled (or first available variant) for each product, low-pass
    for prod, r in results.items():
        ds_R = xr.open_dataset(r["phase2_cache"])
        for try_v in ("R_pooled", "R_sst", "R_ssh"):
            if try_v in ds_R.data_vars:
                R_var = ds_R[try_v]
                break
        # Restrict to RAPID overlap, low-pass, standardise
        R_yymm = np.array([str(t)[:7] for t in R_var.time.values])
        P_yymm = np.array([str(t)[:7] for t in rapid.time.values])
        common_yymm = np.intersect1d(R_yymm, P_yymm)
        if common_yymm.size < 24: continue
        idx_R = np.array([i for i, ym in enumerate(R_yymm) if ym in set(common_yymm)])
        R_o = R_var.isel(time=idx_R).astype(np.float64)
        R_lp = standardise(lowpass_1yr(R_o)).values
        a.plot(R_o.time.values, R_lp, lw=1.2, label=f"{prod} ({try_v})")
        ds_R.close()
    # RAPID
    P_yymm = np.array([str(t)[:7] for t in rapid.time.values])
    common_all = np.intersect1d(np.array([str(t)[:7] for t in rapid.time.values]), P_yymm)
    P_o = rapid.astype(np.float64)
    P_lp = standardise(lowpass_1yr(P_o)).values
    a.plot(rapid.time.values, P_lp, color="black", lw=1.4, label="RAPID 26.5N")
    a.axhline(0, color="grey", lw=0.4)
    a.set_xlabel("year"); a.set_ylabel("standardised")
    a.set_title("(b) R(t) per product vs RAPID, std., low-pass 1 yr")
    a.legend(loc="best", fontsize=8)

    fig.suptitle("Multi-reanalysis H1 test on the Argo-era RAPID overlap",
                 fontsize=11)
    out_fig = MANUSCRIPT_FIGS / "fig1_multi_product.pdf"
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()

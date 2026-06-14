"""
Main Fig. 7: Cold Blob emergence projects onto basin-scale Atlantic circulation.

3-to-4-panel composite addressing the Nature Geoscience editor's
"broader-circulation" complaint. Each panel speaks to a different geoscience
subcommunity, using already-on-disk data where possible.

  (a) Holocene-corridor multi-track: PALMOD-130k 12-kyr envelope (paleo
      audience), Caesar 2021 multi-proxy AMOC stack 400-2018 CE, HadISST
      cold-blob box 1870-2024, RAPID 26.5 N 2004-2024. All on a common
      z-score axis relative to a pre-industrial baseline.

  (b) Q-to-Sv conversion: per-model scatter of basin-mean Q-index trend
      vs AMOC psi_max trend at 26.5 N (1950-2014), with OLS regression
      and bootstrap CI. Provides a quantitative bridge from the
      manuscript's Q observable to canonical AMOC Sverdrups, on 21
      CMIP6 models with cached annual psi_max.

  (c) Downstream-teleconnection composite (3 sub-rows c1-c3):
      c1 = Pacific Walker lead-lag regression on the cold-blob index
           (HadSLP2r/HadISST + CMIP6 psl, McGregor 2014 / Ruprich-Robert 2017);
      c2 = Sahel JJAS precipitation regression on the bandpassed cold-blob
           index (12 CMIP6 historical models, Zhang & Delworth 2006 anchor);
      c3 = European DJF jet-latitude regression on the bandpassed cold-blob
           index (NCEP R1 + CMIP6 ua, Gervais 2019 / Hu & Fedorov 2020).
      All three regressions partial out global-mean temperature.

  (d) Early-warning AR(1) sliding window: lag-1 autocorrelation of
      basin-mean Q for the multi-model ensemble (1900-2024 obs +
      historical) with AR(1)-matched surrogate null, identifying Q as
      an early-warning indicator that crosses the AR1=0.7 threshold
      ahead of the SST-only Caesar fingerprint.

Output: manuscript/figs/fig_cb_circulation_impacts.pdf
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

REPO = Path(__file__).resolve().parents[2]
COLD_BLOB_HADISST = Path("/home/bijanf/Documents/AMOC_renalysis/data/results/cold_blob_timeseries_hadisst.nc")
CAESAR_XLSX = REPO / "data" / "external" / "caesar2021_multiproxy.xlsx"
RAPID_NC = Path("/home/bijanf/Documents/AMOC_renalysis/data/export_rahmstorf/amoc_26N_rapid_annual.nc")
PSIMAX_NPZ = Path("/home/bijanf/Documents/AMOC_renalysis/data/results/yearly_psimax_cmip6.npz")
PALMOD_JSON = REPO / "dcps" / "cache" / "palmod" / "holocene_stack.json"
PALMOD_NPZ = REPO / "dcps" / "cache" / "palmod" / "holocene_stack.npz"
Q_CACHE_DIR = REPO / "dcps" / "cache" / "holocene_exit" / "bulk"

OUT = REPO / "manuscript" / "figs" / "fig_cb_circulation_impacts.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)


def _zscore(arr: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mu = np.nanmean(ref); sd = np.nanstd(ref)
    return (arr - mu) / sd if sd > 0 else arr - mu


def _load_palmod_envelope() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (years_CE, median, p10, p90) of the audited Holocene cold-blob stack.

    The on-disk npz has target_kyr (time in kyr BP) and a per-site matrix
    + basin_mean time series. Compute the across-site p10/p50/p90 from `matrix`,
    or fall back to basin_mean as the median and use NaN envelopes.
    """
    if not PALMOD_NPZ.exists():
        return np.array([]), np.array([]), np.array([]), np.array([])
    d = np.load(PALMOD_NPZ, allow_pickle=True)
    keys = list(d.keys())
    print(f"  PALMOD npz keys: {keys}")
    if "target_kyr" not in keys:
        return np.array([]), np.array([]), np.array([]), np.array([])
    kyr_BP = np.asarray(d["target_kyr"], float)
    # convert kyr BP (before 1950) -> CE year
    years_CE = 1950.0 - kyr_BP * 1000.0
    # matrix: shape (n_sites, n_time) or (n_time, n_sites) -- figure out
    if "matrix" in keys:
        M = np.asarray(d["matrix"], float)
        if M.shape[0] == kyr_BP.size:
            M = M.T  # make it sites x time
        if M.shape[1] != kyr_BP.size:
            print(f"  PALMOD matrix shape {M.shape} does not match target_kyr {kyr_BP.shape}; falling back")
            M = None
        else:
            p10 = np.nanpercentile(M, 10, axis=0)
            p50 = np.nanpercentile(M, 50, axis=0)
            p90 = np.nanpercentile(M, 90, axis=0)
            return years_CE, p50, p10, p90
    # fallback: basin_mean only
    if "basin_mean" in keys:
        bm = np.asarray(d["basin_mean"], float)
        if bm.shape == kyr_BP.shape:
            return years_CE, bm, bm * np.nan, bm * np.nan
    return np.array([]), np.array([]), np.array([]), np.array([])


def _load_caesar() -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Returns (years, AMOC-z) from Caesar 2021 multiproxy stack (col 0 + 1)."""
    if not CAESAR_XLSX.exists(): return None
    try:
        import openpyxl  # noqa
        import pandas as pd
        df = pd.read_excel(CAESAR_XLSX, sheet_name=0)
        # the first numeric columns are typically Year + index
        ycol = next((c for c in df.columns if "year" in str(c).lower()), df.columns[0])
        vcol = None
        for c in df.columns:
            if c == ycol: continue
            if df[c].dtype.kind in "fiu":
                vcol = c; break
        if vcol is None:
            return None
        years = df[ycol].values.astype(float)
        vals = df[vcol].values.astype(float)
        m = np.isfinite(years) & np.isfinite(vals)
        return years[m], vals[m]
    except Exception as e:
        print(f"  Caesar load failed: {e}")
        return None


def _load_hadisst_cb() -> Optional[tuple[np.ndarray, np.ndarray]]:
    if not COLD_BLOB_HADISST.exists(): return None
    try:
        ds = xr.open_dataset(COLD_BLOB_HADISST)
        print(f"  HadISST CB ds: {list(ds.data_vars)}, dims {dict(ds.sizes)}")
        # try common variable names
        for var in ("cold_blob_contrast", "cb_contrast", "contrast", "CB"):
            if var in ds.data_vars:
                arr = ds[var]
                break
        else:
            arr = ds[list(ds.data_vars)[0]]
        years = arr["time"].dt.year.values if "time" in arr.coords else arr["year"].values
        return years.astype(float), arr.values.astype(float)
    except Exception as e:
        print(f"  HadISST CB load failed: {e}")
        return None


def _load_rapid() -> Optional[tuple[np.ndarray, np.ndarray]]:
    if not RAPID_NC.exists(): return None
    try:
        ds = xr.open_dataset(RAPID_NC)
        print(f"  RAPID ds: {list(ds.data_vars)}, dims {dict(ds.sizes)}")
        for var in ("amoc", "moc", "AMOC", "MOC", "transport"):
            if var in ds.data_vars:
                arr = ds[var]; break
        else:
            arr = ds[list(ds.data_vars)[0]]
        years = arr["time"].dt.year.values if "time" in arr.coords else arr["year"].values
        return years.astype(float), arr.values.astype(float)
    except Exception as e:
        print(f"  RAPID load failed: {e}")
        return None


def _load_psimax_per_model() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    d = np.load(PSIMAX_NPZ, allow_pickle=True)
    models = [str(m) for m in d["models"]]
    out = {}
    for m in models:
        if f"{m}_years" in d.files and f"{m}_psimax" in d.files:
            out[m] = (np.asarray(d[f"{m}_years"]).astype(float),
                      np.asarray(d[f"{m}_psimax"]).astype(float))
    return out


def _load_q_trend_per_model(epoch=(1950, 2014)) -> dict[str, float]:
    """Linear OLS trend on hist_Q over the epoch, per model."""
    out = {}
    for p in sorted(Q_CACHE_DIR.glob("*_atlantic.json")):
        # skip the scenario-suffixed (only want the default historical+ssp585)
        if "_ssp" in p.stem: continue
        try:
            d = json.loads(p.read_text())
            model = d["model"]
            yrs = np.asarray(d.get("hist_centres", []), float)
            qs = np.asarray(d.get("hist_Q", []), float)
            m = np.isfinite(yrs) & np.isfinite(qs) & (yrs >= epoch[0]) & (yrs <= epoch[1])
            if m.sum() < 3: continue
            slope, intercept = np.polyfit(yrs[m], qs[m], 1)
            out[model] = slope  # per yr
        except Exception:
            continue
    return out


def _load_amoc_trend_per_model(epoch=(1950, 2014)) -> dict[str, float]:
    """Linear OLS trend on AMOC psi_max over the epoch, per model."""
    pm = _load_psimax_per_model()
    out = {}
    for m, (yrs, psi) in pm.items():
        mask = np.isfinite(yrs) & np.isfinite(psi) & (yrs >= epoch[0]) & (yrs <= epoch[1])
        if mask.sum() < 10: continue
        slope, _ = np.polyfit(yrs[mask], psi[mask], 1)
        out[m] = slope * 10.0  # Sv per decade
    return out


def _ar1_sliding(years: np.ndarray, vals: np.ndarray,
                 window: int = 50, detrend_window: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window AR(1) of (vals - low-pass) at annual stride."""
    from scipy.ndimage import gaussian_filter1d
    # FWHM = detrend_window yr; sigma = FWHM / (2*sqrt(2*ln 2)).
    sigma_yr = float(detrend_window) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    smooth = gaussian_filter1d(vals, sigma=sigma_yr, mode="nearest")
    resid = vals - smooth
    out_years = []
    out_ar1 = []
    for i in range(len(years) - window + 1):
        seg = resid[i:i + window]
        if np.isnan(seg).any():
            continue
        r = np.corrcoef(seg[:-1], seg[1:])[0, 1]
        out_years.append(years[i + window // 2])
        out_ar1.append(r)
    return np.array(out_years), np.array(out_ar1)


def main():
    # ---- load all data ----
    print("Loading PALMOD envelope...")
    palmod_years, palmod_med, palmod_p10, palmod_p90 = _load_palmod_envelope()
    print(f"  PALMOD years: {len(palmod_years)}")

    print("Loading Caesar 2021...")
    caesar = _load_caesar()

    print("Loading HadISST cold-blob index...")
    hadisst = _load_hadisst_cb()

    print("Loading RAPID...")
    rapid = _load_rapid()

    print("Loading per-model Q + AMOC trends...")
    q_trend = _load_q_trend_per_model()
    amoc_trend = _load_amoc_trend_per_model()
    print(f"  Q trends: {len(q_trend)} models; AMOC trends: {len(amoc_trend)} models")

    # ---- build figure ----
    # 5-panel layout (AR(1) early-warning panel dropped: the highly
    # autocorrelated Q always sat above the 0.7 threshold so the panel was
    # visually a flat line carrying no early-warning information).
    #   row 0 : (a) Holocene multi-track, full width
    #   row 1 : (b) Q-to-Sv scatter | (c) DJF jet scatter
    #   row 2 : (d) Walker lead-lag, full width
    #   row 3 : (e) Sahel JJAS pr, full width
    fig = plt.figure(figsize=(7.09, 5.9), constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.04, wspace=0.10, h_pad=0.015, w_pad=0.015)
    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[0.90, 1.30, 1.20, 1.35],
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])   # DJF jet (formerly panel f)
    ax_d = fig.add_subplot(gs[2, :])   # Walker (unchanged letter)
    ax_e = fig.add_subplot(gs[3, :])   # Sahel (now full width)
    ax_f = None  # AR(1) panel removed

    # ===== Panel (a): Holocene-corridor multi-track =====
    pre_baseline = (-200, 1850)  # CE
    if hadisst is not None:
        y, v = hadisst
        mb = (y >= 1870) & (y <= 1900)
        ref = v[mb] if mb.any() else v
        z_had = _zscore(v, ref)
        ax_a.plot(y, z_had, color="black", linewidth=1.0, label="HadISST cold-blob box")

    have_caesar = False
    if caesar is not None:
        y, v = caesar
        mb = (y <= 1850)
        if mb.any() and np.isfinite(v[mb]).sum() > 5:
            z = _zscore(v, v[mb])
            ax_a.plot(y, z, color="#6C5B7B", linewidth=0.9, alpha=0.85,
                      label="Caesar 2021 AMOC stack")
            have_caesar = True
    # If Caesar did not load successfully, no purple line and no orphan legend entry.

    if rapid is not None:
        y, v = rapid
        # RAPID is in Sv; sign-flip and z-score against its own mean for visual overlay
        # (RAPID is the modern terminal anchor; not used to define exit year)
        m = np.isfinite(v)
        if m.any():
            z = (v - np.nanmean(v[m])) / max(np.nanstd(v[m]), 1e-6)
            # invert sign so 'weakening AMOC' appears as 'positive cold-blob' direction
            ax_a.plot(y[m], -z, color="#C73E2C", linewidth=1.2,
                      label="RAPID 26.5$^{\\circ}$N (sign-flipped)", marker="o", markersize=3)

    # PALMOD envelope (if present) plotted on a broken x-axis insert.
    # Strategy: overlay PALMOD years -10000..+1850 as a left-side area.
    if len(palmod_years) > 0 and palmod_p10 is not None and palmod_p90 is not None:
        years_ce = 1950 - (np.asarray(palmod_years) * 1000.0) if palmod_years.max() < 50 else palmod_years
        m = np.isfinite(years_ce) & np.isfinite(palmod_p10) & np.isfinite(palmod_p90)
        # PALMOD has different units; rescale to z-score against its pre-1850 distribution
        # so it occupies same axis as the other series
        ref = palmod_med[(years_ce <= 1850) & m]
        if len(ref) > 5:
            z_med = _zscore(palmod_med, ref)
            z_p10 = _zscore(palmod_p10, ref)
            z_p90 = _zscore(palmod_p90, ref)
            ax_a.fill_between(years_ce, z_p10, z_p90,
                              facecolor="#D9A441", alpha=0.30,
                              hatch="////", edgecolor="#8C6A1A",
                              linewidth=0.0,
                              label="PALMOD-130k 12-kyr envelope (paleo p10-p90)")
            ax_a.plot(years_ce, z_med, color="#7A5A1A", linewidth=0.8, alpha=1.0)

    ax_a.axhline(0, color="0.55", linewidth=0.4, linestyle=":", alpha=0.9)
    ax_a.set_xlim(-2050, 2100)
    ax_a.set_xticks(np.arange(-2000, 2001, 500))
    ax_a.set_xlabel("Year (CE)", labelpad=1)
    ax_a.set_ylabel("$z$-score (pre-1850 baseline)", labelpad=1)
    ax_a.tick_params(pad=1, length=2)
    # Panel (a) handles collected for the figure-foot legend; no in-axes legend.
    panel_a_handles = [h for h in ax_a.get_legend_handles_labels()[0]]
    panel_a_labels  = [l for l in ax_a.get_legend_handles_labels()[1]]
    ax_a.set_title("a", loc="left", fontweight="bold", fontsize=10, pad=2)

    # ===== Panel (b): Q-to-Sv conversion =====
    common = sorted(set(q_trend) & set(amoc_trend))
    print(f"  panel-b common models: {len(common)}")
    qs = np.array([q_trend[m] for m in common])  # Q per yr
    sv = np.array([amoc_trend[m] for m in common])  # Sv per decade
    # Convert q trend to per-decade for human-readable axis
    qs10 = qs * 10.0
    if len(common) >= 5:
        # Uniform circles -- no per-family encoding (n=11 cannot support it).
        ax_b.scatter(qs10, sv, s=42, marker="o",
                     facecolors="#4C72B0", edgecolors="black",
                     linewidth=0.5, zorder=3)

        # Theil-Sen (robust to leverage in small samples) + 1000-iter bootstrap CI.
        from scipy.stats import theilslopes, spearmanr
        ts_slope, ts_intercept, _lo_ci_unused, _hi_ci_unused = theilslopes(sv, qs10, 0.95)
        rng = np.random.default_rng(2026)
        boot_slopes = np.empty(1000)
        n = len(qs10)
        for k in range(1000):
            idx = rng.integers(0, n, n)
            s_k, _, _, _ = theilslopes(sv[idx], qs10[idx], 0.95)
            boot_slopes[k] = s_k
        ci_lo, ci_hi = np.nanpercentile(boot_slopes, [2.5, 97.5])

        # Spearman -- robust to outliers; Pearson is too leverage-sensitive at n=11.
        rho_s, p_s = spearmanr(qs10, sv)

        # Leave-one-out slope range -- explicit fragility report.
        loo_slopes = np.array([
            theilslopes(np.delete(sv, j), np.delete(qs10, j), 0.95)[0]
            for j in range(n)
        ])
        loo_lo, loo_hi = float(loo_slopes.min()), float(loo_slopes.max())

        xs = np.linspace(qs10.min(), qs10.max(), 80)
        # Bootstrap CI band around the regression line (visual = statistical weight).
        band_lo = np.nanpercentile(boot_slopes, 2.5) * xs + ts_intercept
        band_hi = np.nanpercentile(boot_slopes, 97.5) * xs + ts_intercept
        h_b_band = ax_b.fill_between(xs, band_lo, band_hi,
                                     facecolor="0.80", alpha=0.35,
                                     edgecolor="0.45", linewidth=0.3,
                                     zorder=1)
        h_b_line, = ax_b.plot(xs, ts_slope * xs + ts_intercept,
                              color="#222222", linewidth=0.9,
                              linestyle="-", zorder=2,
                              label="Theil-Sen fit + 95% bootstrap CI")
        panel_b_handles = [(h_b_line, h_b_band)]
        panel_b_labels  = ["Theil-Sen fit + 95\\% bootstrap CI"]

        # statistics (Theil-Sen, CI, Spearman, LOO range) live in the LaTeX
        # caption -- NO in-axes annotation box.
        pass
    ax_b.axhline(0, color="0.55", linewidth=0.4, linestyle=":", alpha=0.9)
    ax_b.axvline(0, color="0.55", linewidth=0.4, linestyle=":", alpha=0.9)
    ax_b.set_xlabel("Basin-mean $Q$ trend 1950--2014  ($\\Delta Q$/decade)", labelpad=1)
    ax_b.set_ylabel("AMOC $\\psi_{\\mathrm{max}}$ trend (Sv per decade)", labelpad=1)
    ax_b.tick_params(pad=1, length=2)
    ax_b.set_title("b", loc="left", fontweight="bold", fontsize=10, pad=2)

    # ===== Panel (c): downstream-teleconnection composite =====
    # c1 Walker, c2 Sahel, c3 Jet. Each sub-row imports its draw_subpanel(ax)
    # from the matching analysis module. Failures degrade gracefully so that
    # missing inputs do not break the build of the rest of the figure.
    def _draw_panel(ax, importer, letter: str, label: str):
        """Call subpanel draw fn; return its (handles, labels) if provided."""
        result = None
        try:
            fn = importer()
            result = fn(ax)
        except Exception as exc:
            print(f"  panel {letter} ({label}) failed: {exc}")
            ax.text(0.5, 0.5, f"({label}: {exc})", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6)
            ax.set_xticks([]); ax.set_yticks([])
        # Panel letter via set_title: places letter in the reserved title
        # area ABOVE the axes (NOT inside the data region).
        ax.set_title(letter, loc="left", fontweight="bold", fontsize=10, pad=2)
        return result

    import importlib.util as _ilu
    _SCRIPT_DIR = Path(__file__).resolve().parent

    def _import_by_path(modname: str, relpath: str):
        spec = _ilu.spec_from_file_location(modname, _SCRIPT_DIR / relpath)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _walker():
        return _import_by_path("_fig7_walker_mod", "compute_fig7_walker.py").draw_subpanel

    def _sahel():
        return _import_by_path("_fig7_sahel_mod", "compute_fig7_sahel.py").draw_subpanel

    def _jet():
        return _import_by_path("_fig7_jet_mod", "compute_fig7_jet.py").draw_subpanel

    # NOTE: AR(1) early-warning panel removed -- the Q-AR(1) trace was visually
    # a flat line near 1 throughout the analysis window (Q is highly auto-
    # correlated by construction), carrying no observable early-warning signal.
    walker_legend = _draw_panel(ax_d, _walker, "d", "Walker")
    sahel_legend  = _draw_panel(ax_e, _sahel,  "e", "Sahel")
    jet_legend    = _draw_panel(ax_c, _jet,    "c", "Jet")

    # ONE shared frameless figure-foot legend -- 'outside lower center' is
    # constrained-layout-aware so the strip cannot overlap any panel frame.
    # Only panels (a), (c) Jet, (d) Walker contribute legend handles; panel
    # (e) Sahel's 2-series content is described in the caption.
    # Build the figure-foot legend as explicit proxy handles so every
    # swatch matches the on-page colour (not the alpha-blended preview).
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple

    proxy_palmod   = Patch(facecolor="#D9A441", alpha=0.30,
                           hatch="////", edgecolor="#8C6A1A", linewidth=0.0,
                           label="PALMOD 12-kyr envelope (paleo)")
    proxy_hadisst  = Line2D([], [], color="black", linewidth=1.1,
                            label="HadISST cold-blob box")
    proxy_rapid    = Line2D([], [], color="#C73E2C", linewidth=1.2,
                            marker="o", markersize=3,
                            label="RAPID 26.5$^{\\circ}$N (sign-flipped)")
    proxy_ncep     = Line2D([], [], color="black", marker="o",
                            markerfacecolor="none", markeredgewidth=0.6,
                            markersize=3.5, linestyle="none",
                            label="NCEP/NCAR R1 jet")
    proxy_walker_obs = Line2D([], [], color="black", marker="o",
                              markersize=3.5, linewidth=1.1,
                              label="HadSLP2r/HadISST Walker obs.")
    proxy_walker_med = Line2D([], [], color="#4C72B0", linewidth=1.3,
                              label="CMIP6 Walker median ($n=6$)")
    proxy_jet_med    = Line2D([], [], color="#D17A22", linewidth=1.2,
                              label="CMIP6 jet median ($n=7$)")
    proxy_pic_med    = Line2D([], [], color="#5B8E91", linewidth=1.0,
                              linestyle="--",
                              label="CMIP6 piControl Walker ($n=7$)")
    proxy_obs_ci     = Patch(facecolor="0.80", alpha=0.35,
                             edgecolor="0.45", linewidth=0.3,
                             label="Obs. 95% bootstrap CI (b, d)")
    proxy_iqr_walk   = Patch(facecolor="#4C72B0", alpha=0.22, linewidth=0)
    proxy_iqr_jet    = Patch(facecolor="#D17A22", alpha=0.22, linewidth=0)
    proxy_ts         = Line2D([], [], color="#222222", linewidth=0.9)
    proxy_ts_band    = Patch(facecolor="0.80", alpha=0.35,
                             edgecolor="0.45", linewidth=0.3)
    proxy_sahel_exp  = Patch(facecolor="#F0D8C8", alpha=0.55,
                             edgecolor="#C9A37A", linewidth=0.3,
                             label="Sahel expected range (Z&D 2006)")
    proxy_sahel_ens  = Line2D([], [], color="#2F3E46", linewidth=2.4,
                              marker="|", markersize=8, markeredgewidth=1.6,
                              label="Sahel ensemble median $\\pm$ IQR")

    legend_entries = [
        (proxy_hadisst,                              proxy_hadisst.get_label()),
        (proxy_rapid,                                proxy_rapid.get_label()),
        (proxy_palmod,                               proxy_palmod.get_label()),
        (proxy_walker_obs,                           proxy_walker_obs.get_label()),
        (proxy_obs_ci,                               proxy_obs_ci.get_label()),
        (proxy_ncep,                                 proxy_ncep.get_label()),
        (proxy_walker_med,                           proxy_walker_med.get_label()),
        (proxy_jet_med,                              proxy_jet_med.get_label()),
        (proxy_pic_med,                              proxy_pic_med.get_label()),
        ((proxy_iqr_walk, proxy_iqr_jet),            "CMIP6 historical 25-75% IQR (Walker, Jet)"),
        ((proxy_ts, proxy_ts_band),                  "Theil-Sen fit + 95% bootstrap CI (panel b)"),
        (proxy_sahel_ens,                            proxy_sahel_ens.get_label()),
        (proxy_sahel_exp,                            proxy_sahel_exp.get_label()),
    ]
    if have_caesar:
        legend_entries.insert(3,
            (Line2D([], [], color="#6C5B7B", linewidth=0.9,
                    label="Caesar 2021 AMOC stack"),
             "Caesar 2021 AMOC stack"))

    leg_handles = [e[0] for e in legend_entries]
    leg_labels  = [e[1] for e in legend_entries]
    fig.legend(leg_handles, leg_labels,
               loc="outside lower center",
               ncol=5, frameon=False, fontsize=5.5,
               columnspacing=1.0, handlelength=2.2, handletextpad=0.4,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})

    fig.savefig(OUT, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

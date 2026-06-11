"""
Compute the ratio N_v50 / N_v53 of SN Ia survivors after selection cuts
for LSST OpSim baselines v5.0.1 and v5.3.0.

Usage:
    # single cut combo (CLI args)
    python snia_selection_ratio.py [--snr-threshold 5.0] [--n-phases 10] ...

    # sweep over many cut combos without re-running the simulation
    python snia_selection_ratio.py --cuts-file cuts.json [--verbose]

cuts.json format — list of objects, each overriding any subset of the 7 cut params:
    [
      {"snr_threshold": 5.0, "n_phases": 10},
      {"snr_threshold": 3.0, "n_phases": 10},
      {"snr_threshold": 5.0, "n_phases": 7, "n_before_peak": 1}
    ]
"""

import argparse
import functools
import json
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
from scipy.interpolate import interp1d

from lightcurvelynx import _LIGHTCURVELYNX_DOWNLOAD_DATA_DIR
from lightcurvelynx.astro_utils.dustmap import SFDMap
from lightcurvelynx.astro_utils.passbands import PassbandGroup
from lightcurvelynx.astro_utils.snia_utils import (
    DistModFromRedshift,
    X0FromDistMod,
    num_snia_per_redshift_bin,
    snia_volumetric_rates,
)
from lightcurvelynx.effects.extinction import ExtinctionEffect
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.math_nodes.ra_dec_sampler import ApproximateMOCSampler
from lightcurvelynx.math_nodes.scipy_random import SamplePDF
from lightcurvelynx.models.sncosmo_models import SncosmoWrapperModel
from lightcurvelynx.obstable.opsim import OpSim
from lightcurvelynx.simulate import simulate_lightcurves
from lightcurvelynx.utils.extrapolate import LinearDecayOnMag, ZeroPadding

_OPSIM_SQL = "SELECT * FROM observations WHERE scheduler_note NOT LIKE 'DD:%'"

SIM_PARAMS = {
    "H0": 70.0,
    "Omega_m": 0.315,
    "w": -1.0,
    "zmin": 0.001,
    "zmax": 1.0,
    "znbins": 100,
    "alpha": 0.15,
    "beta": 3.15,
    "x1_mean": 0.973,
    "x1_sigma_minus": 1.472,
    "x1_sigma_plus": 0.222,
    "c_mean": -0.054,
    "c_sigma_minus": 0.043,
    "c_sigma_plus": 0.101,
    "m_abs_mean": -19.3,
    "m_abs_sigma": 0.1,
    "filters": ["u", "g", "r", "i", "z", "y"],
}


def _asymmetric_gaussian_pdf(x, mu, sigma_minus, sigma_plus):
    norm_factor = np.sqrt(2 / np.pi) / (sigma_minus + sigma_plus)
    return np.where(
        x < mu,
        norm_factor * np.exp(-0.5 * ((x - mu) / sigma_minus) ** 2),
        norm_factor * np.exp(-0.5 * ((x - mu) / sigma_plus) ** 2),
    )


def _lc_quality_cuts(flux, mjd, filter, z, t0,
                     n_phases, phase_min, phase_max,
                     n_before_peak, n_after_peak, n_bands):
    phases = np.floor((mjd - t0) / (1.0 + z))
    unique_phases, unique_idx = np.unique(phases, return_index=True)
    good_idx = (unique_phases >= phase_min) & (unique_phases <= phase_max)
    if np.sum(good_idx) == 0:
        return {"pass_quality_cuts": False}
    pass_cut = len(unique_phases[good_idx]) >= n_phases
    pass_cut &= np.sum(unique_phases[good_idx] < 0) >= n_before_peak
    pass_cut &= np.sum(unique_phases[good_idx] > 0) >= n_after_peak
    pass_cut &= len(np.unique(filter[unique_idx][good_idx])) >= n_bands
    return {"pass_quality_cuts": pass_cut}


def run_simulation(opsim_path, seed, fraction, verbose):
    """Simulate light curves; return raw results DataFrame (no cuts applied)."""
    rng = np.random.default_rng(seed)
    p = SIM_PARAMS

    opsim = OpSim.from_db(opsim_path, sql_query=_OPSIM_SQL)
    if verbose:
        print(f"  OpSim: {len(opsim):,} observations  "
              f"(MJD {opsim['time'].min():.1f} – {opsim['time'].max():.1f})")

    sky_coverage = opsim.estimate_coverage()
    if verbose:
        print(f"  Sky coverage: {sky_coverage:.0f} deg²")

    t_min = float(opsim["time"].min())
    t_max = float(opsim["time"].max())
    survey_length = (t_max - t_min) / 365.25
    solid_angle = sky_coverage * (np.pi / 180.0) ** 2

    nsntotal, _ = num_snia_per_redshift_bin(
        p["zmin"], p["zmax"],
        znbins=1,
        solid_angle=solid_angle,
        vol_rate_function=snia_volumetric_rates,
        H0=p["H0"],
        Omega_m=p["Omega_m"],
    )
    nsn = int(int(nsntotal[0] * survey_length) * fraction)
    if verbose:
        print(f"  Simulating {nsn:,} SNe Ia ({fraction*100:.1f}% of expected)")

    passbands = PassbandGroup.from_preset("LSST", filters=p["filters"])

    nsn_per_bin, z_mean = num_snia_per_redshift_bin(
        p["zmin"], p["zmax"], p["znbins"],
        H0=p["H0"], Omega_m=p["Omega_m"],
    )
    zpdf = interp1d(z_mean, nsn_per_bin, bounds_error=False, fill_value=0)

    moc = opsim.build_moc(max_depth=12)
    radec = ApproximateMOCSampler(moc, node_label="radec")
    z_func = SamplePDF(zpdf, node_label="redshift")

    def x1_pdf(x):
        return _asymmetric_gaussian_pdf(
            x, p["x1_mean"], p["x1_sigma_minus"], p["x1_sigma_plus"]
        )

    def c_pdf(c):
        return _asymmetric_gaussian_pdf(
            c, p["c_mean"], p["c_sigma_minus"], p["c_sigma_plus"]
        )

    x1_func = SamplePDF(x1_pdf, node_label="x1")
    c_func = SamplePDF(c_pdf, node_label="c")
    m_abs_func = NumpyRandomFunc("normal", loc=p["m_abs_mean"], scale=p["m_abs_sigma"])
    distmod_func = DistModFromRedshift(z_func, H0=p["H0"], Omega_m=p["Omega_m"])
    x0_func = X0FromDistMod(
        distmod=distmod_func,
        x1=x1_func,
        c=c_func,
        alpha=p["alpha"],
        beta=p["beta"],
        m_abs=m_abs_func,
        node_label="x0_func",
    )

    source = SncosmoWrapperModel(
        "salt3",
        t0=NumpyRandomFunc("uniform", low=t_min, high=t_max),
        x0=x0_func,
        x1=x1_func,
        c=c_func,
        ra=radec.ra,
        dec=radec.dec,
        redshift=z_func,
        node_label="source",
        time_extrapolation=(ZeroPadding(), LinearDecayOnMag(decay_rate=0.02, mag_thres=30.0)),
        wave_extrapolation=(ZeroPadding(), ZeroPadding()),
    )
    mwextinction = SFDMap(ra=source.ra, dec=source.dec, node_label="mwext")
    source.add_effect(
        ExtinctionEffect(
            extinction_model="F99", ebv=mwextinction,
            r_v=3.1, frame="observer", backend="dust_extinction",
        )
    )

    param_cols = [
        "source.t0", "source.x0", "source.x1", "source.c",
        "source.redshift", "source.ra", "source.dec", "x0_func.distmod",
    ]
    try:
        import loky
        executor = loky.get_reusable_executor(max_workers=4)
    except ImportError:
        executor = None

    results = simulate_lightcurves(
        model=source,
        num_samples=nsn,
        survey_info=opsim,
        passbands=passbands,
        param_cols=param_cols,
        obstable_save_cols=["zp"],
        rng=rng,
        num_jobs=4,
        batch_size=2000,
        executor=executor,
    )

    if verbose:
        print(f"  Simulated {len(results):,} SNe Ia")

    return results


def apply_cuts(results, cut_params, verbose=False):
    """Apply selection cuts to raw simulation results; return survivor count."""
    lightcurves = results.dropna(subset=["lightcurve"])
    lightcurves["lightcurve.snr"] = (
        lightcurves["lightcurve.flux"] / lightcurves["lightcurve.fluxerr"]
    )
    lightcurves["lightcurve.detection_flag"] = (
        lightcurves["lightcurve.snr"] > cut_params["snr_threshold"]
    )

    lightcurves_after_drop_sat = lightcurves.query(
        "lightcurve.is_saturated == False"
    ).dropna(subset=["lightcurve"])

    n_before_det = len(lightcurves)
    lightcurves_after_detection = lightcurves_after_drop_sat.query(
        "lightcurve.detection_flag == True"
    ).dropna(subset=["lightcurve"])
    n_after_det = len(lightcurves_after_detection)

    quality_cut_fn = functools.partial(
        _lc_quality_cuts,
        n_phases=cut_params["n_phases"],
        phase_min=cut_params["phase_min"],
        phase_max=cut_params["phase_max"],
        n_before_peak=cut_params["n_before_peak"],
        n_after_peak=cut_params["n_after_peak"],
        n_bands=cut_params["n_bands"],
    )
    pass_quality_cut = lightcurves_after_detection.reduce(
        quality_cut_fn,
        "lightcurve.flux", "lightcurve.mjd", "lightcurve.filter", "z", "t0",
    )
    idx = pass_quality_cut.query("pass_quality_cuts == True").index
    n_after_quality = len(idx)

    if verbose:
        print(f"    Before detection cut: {n_before_det:,}")
        print(f"    After  detection cut: {n_after_det:,}")
        print(f"    After  quality  cuts: {n_after_quality:,}")

    return n_after_quality


def _row_label(cp):
    return cp.get("label", _fmt_cut_params(cp))


def save_summary_plot(rows, path):
    labels = [_row_label(cp) for cp, *_ in rows]
    n_v50 = [r[1] for r in rows]
    n_v53 = [r[2] for r in rows]
    ratios = [r[3] for r in rows]

    x = range(len(rows))
    width = 0.35

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(8, len(rows) * 1.2), 8),
                                         sharex=True)
    fig.subplots_adjust(hspace=0.08)

    bars1 = ax_top.bar([i - width / 2 for i in x], n_v50, width, label="v5.0.1", color="steelblue")
    bars2 = ax_top.bar([i + width / 2 for i in x], n_v53, width, label="v5.3.0", color="darkorange")
    ax_top.set_ylabel("SNe Ia surviving cuts")
    ax_top.legend()
    ax_top.bar_label(bars1, fmt="%d", fontsize=8, padding=2)
    ax_top.bar_label(bars2, fmt="%d", fontsize=8, padding=2)

    ax_bot.bar(x, ratios, color="steelblue", alpha=0.7)
    ax_bot.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax_bot.set_ylabel("N_v50 / N_v53")
    ax_bot.set_xticks(list(x))
    ax_bot.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

    fig.suptitle("SN Ia yield comparison: OpSim v5.0.1 vs v5.3.0", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {path}")


def _fmt_cut_params(cp):
    return (
        f"snr={cp['snr_threshold']} phases={cp['n_phases']} "
        f"bef={cp['n_before_peak']} aft={cp['n_after_peak']} "
        f"bands={cp['n_bands']} [{cp['phase_min']},{cp['phase_max']}]"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute N_v50 / N_v53 SN Ia yield ratio for two OpSim baselines."
    )
    # Selection cut defaults (used when no --cuts-file, or as fallback values)
    parser.add_argument("--snr-threshold", type=float, default=5.0,
                        help="SNR detection threshold (default: 5.0)")
    parser.add_argument("--n-phases", type=int, default=10,
                        help="Min unique phases in window (default: 10)")
    parser.add_argument("--phase-min", type=int, default=-10,
                        help="Phase window start in rest-frame days (default: -10)")
    parser.add_argument("--phase-max", type=int, default=40,
                        help="Phase window end in rest-frame days (default: 40)")
    parser.add_argument("--n-before-peak", type=int, default=2,
                        help="Min pre-peak observations (default: 2)")
    parser.add_argument("--n-after-peak", type=int, default=3,
                        help="Min post-peak observations (default: 3)")
    parser.add_argument("--n-bands", type=int, default=2,
                        help="Min distinct bands (default: 2)")
    # Batch mode
    parser.add_argument("--cuts-file", type=Path, default=None,
                        help="JSON file with list of cut-parameter objects to sweep")
    # Simulation options
    parser.add_argument("--fraction", type=float, default=0.002,
                        help="Fraction of expected SNe to simulate (default: 0.002)")
    parser.add_argument("--seed", type=int, default=1024,
                        help="Random seed (default: 1024)")
    parser.add_argument(
        "--opsim-v50",
        type=Path,
        default=_LIGHTCURVELYNX_DOWNLOAD_DATA_DIR / "opsim" / "baseline_v5.0.1_10yrs.db",
        help="Path to OpSim v5.0.1 database",
    )
    parser.add_argument(
        "--opsim-v53",
        type=Path,
        default=_LIGHTCURVELYNX_DOWNLOAD_DATA_DIR / "opsim" / "baseline_v5.3.0_10yrs.db",
        help="Path to OpSim v5.3.0 database",
    )
    parser.add_argument("--plot", type=Path, default=None,
                        help="Save summary plot to this path (e.g. selection_ratio.png)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-step counts")

    args = parser.parse_args()

    default_cut_params = {
        "snr_threshold": args.snr_threshold,
        "n_phases": args.n_phases,
        "phase_min": args.phase_min,
        "phase_max": args.phase_max,
        "n_before_peak": args.n_before_peak,
        "n_after_peak": args.n_after_peak,
        "n_bands": args.n_bands,
    }

    if args.cuts_file is not None:
        overrides = json.loads(args.cuts_file.read_text())
        cut_combos = [{**default_cut_params, **o} for o in overrides]
    else:
        cut_combos = [default_cut_params]

    # Simulate once per OpSim version
    if args.verbose:
        print("=== Simulating v5.0.1 ===")
    results_v50 = run_simulation(args.opsim_v50, args.seed, args.fraction, args.verbose)

    if args.verbose:
        print("=== Simulating v5.3.0 ===")
    results_v53 = run_simulation(args.opsim_v53, args.seed, args.fraction, args.verbose)

    # Apply each cut combo
    rows = []
    for cp in cut_combos:
        if args.verbose:
            print(f"--- cuts: {_fmt_cut_params(cp)} ---")
        n_v50 = apply_cuts(results_v50, cp, verbose=args.verbose)
        n_v53 = apply_cuts(results_v53, cp, verbose=args.verbose)
        ratio = n_v50 / n_v53 if n_v53 > 0 else float("nan")
        rows.append((cp, n_v50, n_v53, ratio))

    if len(rows) == 1:
        cp, n_v50, n_v53, ratio = rows[0]
        print(f"ratio = N_v50 / N_v53 = {ratio:.4f}  (N_v50={n_v50:,}, N_v53={n_v53:,})")
    else:
        col_w = max(len(_row_label(r[0])) for r in rows)
        header = f"{'label':<{col_w}}  {'N_v50':>7}  {'N_v53':>7}  {'ratio':>8}"
        print(header)
        print("-" * len(header))
        for cp, n_v50, n_v53, ratio in rows:
            print(f"{_row_label(cp):<{col_w}}  {n_v50:>7,}  {n_v53:>7,}  {ratio:>8.4f}")

    if args.plot is not None:
        save_summary_plot(rows, args.plot)


if __name__ == "__main__":
    main()

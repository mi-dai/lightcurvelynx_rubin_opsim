"""
Compute the ratio N_v50 / N_v53 of kilonova survivors after selection cuts
for LSST OpSim baselines v5.0.1 and v5.3.0.

Usage:
    # single cut combo (CLI args)
    python kn_selection_ratio.py [--snr-threshold 5.0] [--n-phases 5] ...

    # sweep over many cut combos without re-running the simulation
    python kn_selection_ratio.py --cuts-file cuts.json [--verbose]

cuts.json format — list of objects, each overriding any subset of the 7 cut params:
    [
      {"snr_threshold": 5.0, "n_phases": 5},
      {"snr_threshold": 3.0, "n_phases": 5},
      {"snr_threshold": 5.0, "n_phases": 3, "n_after_peak": 2}
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
from lightcurvelynx.astro_utils.snia_utils import num_snia_per_redshift_bin
from lightcurvelynx.effects.extinction import ExtinctionEffect
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.math_nodes.ra_dec_sampler import ApproximateMOCSampler
from lightcurvelynx.models.redback_models import RedbackWrapperModel
from lightcurvelynx.obstable.opsim import OpSim
from lightcurvelynx.simulate import simulate_lightcurves
from lightcurvelynx.utils.extrapolate import LinearDecayOnMag, ZeroPadding

from redback import model_library

_OPSIM_SQL = "SELECT * FROM observations WHERE scheduler_note NOT LIKE 'DD:%'"
_KN_RATE = 1.0e-6  # yr^-1 Mpc^-3 (Scolnic et al. 2018)

SIM_PARAMS = {
    "H0": 70.0,
    "Omega_m": 0.315,
    "w": -1.0,
    "zmin": 0.001,
    "zmax": 0.4,
    "znbins": 100,
    "mej_mean": 0.05,
    "mej_sigma": 0.02,
    "temperature_floor": 3000,
    "kappa": 1,
    "vej": 0.2,
    "filters": ["u", "g", "r", "i", "z", "y"],
}


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
    """Simulate KN light curves; return raw results DataFrame (no cuts applied)."""
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

    # N_kn via volumetric rate integral (flat rate = _KN_RATE)
    kn_rate_fn = lambda z: _KN_RATE  # noqa: E731
    nsntotal, _ = num_snia_per_redshift_bin(
        p["zmin"], p["zmax"],
        znbins=1,
        solid_angle=solid_angle,
        vol_rate_function=kn_rate_fn,
        H0=p["H0"],
        Omega_m=p["Omega_m"],
    )
    nsn = int(int(nsntotal[0] * survey_length) * fraction)
    if verbose:
        print(f"  Simulating {nsn:,} KNe ({fraction*100:.1f}% of expected)")

    passbands = PassbandGroup.from_preset("LSST", filters=p["filters"])

    moc = opsim.build_moc(max_depth=12)
    radec = ApproximateMOCSampler(moc, node_label="radec")

    rb_model = model_library.all_models_dict["one_component_kilonova_model"]
    parameters = {
        "mej": NumpyRandomFunc("normal", loc=p["mej_mean"], scale=p["mej_sigma"]),
        "redshift": NumpyRandomFunc("uniform", low=0.0, high=0.1),
        "temperature_floor": p["temperature_floor"],
        "kappa": p["kappa"],
        "vej": p["vej"],
    }

    source = RedbackWrapperModel(
        rb_model,
        parameters=parameters,
        ra=radec.ra,
        dec=radec.dec,
        t0=NumpyRandomFunc("uniform", low=t_min, high=t_max),
        node_label="source",
    )
    mwextinction = SFDMap(ra=source.ra, dec=source.dec, node_label="mwext")
    source.add_effect(
        ExtinctionEffect(
            extinction_model="F99", ebv=mwextinction,
            r_v=3.1, frame="observer", backend="dust_extinction",
        )
    )

    param_cols = [
        "source.redshift",
        "source.ra",
        "source.dec",
        "source.t0",
        "source.mej",
        "source.temperature_floor",
        "source.kappa",
        "source.vej",
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
        rest_time_window_offset=(0.1, 50),
    )

    if verbose:
        print(f"  Simulated {len(results):,} KNe")

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


def _fmt_cut_params(cp):
    return (
        f"snr={cp['snr_threshold']} phases={cp['n_phases']} "
        f"bef={cp['n_before_peak']} aft={cp['n_after_peak']} "
        f"bands={cp['n_bands']} [{cp['phase_min']},{cp['phase_max']}]"
    )


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
    ax_top.set_ylabel("KNe surviving cuts")
    ax_top.legend()
    ax_top.bar_label(bars1, fmt="%d", fontsize=8, padding=2)
    ax_top.bar_label(bars2, fmt="%d", fontsize=8, padding=2)

    ax_bot.bar(x, ratios, color="steelblue", alpha=0.7)
    ax_bot.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax_bot.set_ylabel("N_v50 / N_v53")
    ax_bot.set_xticks(list(x))
    ax_bot.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

    fig.suptitle("KN yield comparison: OpSim v5.0.1 vs v5.3.0", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute N_v50 / N_v53 KN yield ratio for two OpSim baselines."
    )
    # Selection cut defaults
    parser.add_argument("--snr-threshold", type=float, default=5.0,
                        help="SNR detection threshold (default: 5.0)")
    parser.add_argument("--n-phases", type=int, default=5,
                        help="Min unique phases in window (default: 5)")
    parser.add_argument("--phase-min", type=int, default=-10,
                        help="Phase window start in rest-frame days (default: -10)")
    parser.add_argument("--phase-max", type=int, default=40,
                        help="Phase window end in rest-frame days (default: 40)")
    parser.add_argument("--n-before-peak", type=int, default=0,
                        help="Min pre-peak observations (default: 0)")
    parser.add_argument("--n-after-peak", type=int, default=1,
                        help="Min post-peak observations (default: 1)")
    parser.add_argument("--n-bands", type=int, default=1,
                        help="Min distinct bands (default: 1)")
    # Batch mode
    parser.add_argument("--cuts-file", type=Path, default=None,
                        help="JSON file with list of cut-parameter objects to sweep")
    # Simulation options
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of expected KNe to simulate (default: 1.0)")
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
                        help="Save summary plot to this path (e.g. kn_ratio.png)")
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

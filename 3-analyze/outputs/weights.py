#!/usr/bin/env python
import json
import os
import sys
import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pylab as pl
from scipy.stats import linregress
from tqdm import tqdm

from eos_utils.eosfit_31_adapted import BM
from quantities_for_comparison import (
    delta, nu, epsilon, get_volume_scaling_to_formula_unit
)

# --- Matplotlib and LaTeX Font Configuration ---
pl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

# --- Fit Helper ---
def fit_bm3(volumes: np.ndarray, energies: np.ndarray) -> tuple[float, float, float, float]:
    """Fit BM3 and return (E0, V0, B0, B1). Energies in eV/cell. V in Å^3/cell."""
    if len(volumes) < 4:
        raise ValueError("Need at least 4 points to fit a 3rd-order polynomial.")
    try:
        v0, e0, b0_internal, bp, _ = BM(np.column_stack((volumes, energies)))
        popt = (e0, v0, b0_internal, bp)
    except (ValueError, np.linalg.LinAlgError):
            raise RuntimeError(f"BM fit failed")
    return popt

# --- Bootstrap Function ---
def bootstrap_param_std(
    volumes: np.ndarray, energies: np.ndarray, n_rep: int, noise_ev: float, use_mad: bool = False
) -> tuple[float, float, float]:
    """Return sigma (or MAD*1.4826) for V0, B0, B1 over bootstrap replicas."""
    n_points = len(volumes)
    replicate_v0, replicate_b0, replicate_b1 = [], [], []
    for _ in range(n_rep):
        idx = np.random.choice(np.arange(n_points), size=n_points, replace=True)
        v_boot = volumes[idx]
        e_boot = energies[idx] + np.random.normal(0.0, noise_ev * 1e-3, n_points)
        try:
            _, v0, b0, b1 = fit_bm3(v_boot, e_boot)
            replicate_v0.append(v0); replicate_b0.append(b0); replicate_b1.append(b1)
        except (RuntimeError, ValueError):
            continue 
    stats = []
    for arr in (replicate_v0, replicate_b0, replicate_b1):
        a = np.asarray(arr)
        if len(a) < 2:
            stats.append(0.0)
            continue
        if use_mad:
            med = np.median(a); mad = np.median(np.abs(a - med))
            stats.append(1.4826 * mad)
        else:
            stats.append(np.std(a, ddof=1))
    return tuple(stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap-based adaptive weighting for the nu-metric", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("set_name", help="Dataset prefix, e.g., oxides-verification-PBE-v1")
    parser.add_argument("--plugin", default="dftk", help="Code/plugin name")
    parser.add_argument("--n-rep", type=int, default=500, help="Number of bootstrap replicas per material")
    parser.add_argument("--noise", type=float, default=1.0, help="Std. dev. of Gaussian noise in meV/cell")
    parser.add_argument("--mad", action="store_true", help="Use Median Absolute Deviation (robust to outliers)")
    args = parser.parse_args()
    
    SET_NAME, PLUGIN_NAME = args.set_name, args.plugin
    input_dir = Path(f"{SET_NAME}-{PLUGIN_NAME}")
    ref_results_file = input_dir / "results_2high_ecut.json"

    try:
        with open(ref_results_file, 'r') as f: ref_data = json.load(f)
    except FileNotFoundError: sys.exit(f"ERROR: High-Ecut reference data file not found at: {ref_results_file}")

    print("--- PART 1: Calculating weights via bootstrap on high-Ecut data ---")
    high_ecut_curves = ref_data.get("eos_data", {})
    if not high_ecut_curves: sys.exit("ERROR: 'eos_data' not found in reference file.")
        
    sig2_v0, sig2_b0, sig2_b1 = [], [], []
    for material_key, eos_points in tqdm(high_ecut_curves.items(), desc="Bootstrapping"):
        if not eos_points or len(eos_points) < 4: continue
        volumes, energies = np.array(eos_points).T
        s_v0, s_b0, s_b1 = bootstrap_param_std(volumes, energies, n_rep=args.n_rep, noise_ev=args.noise, use_mad=args.mad)
        if s_v0 > 0 and s_b0 > 0 and s_b1 > 0:
            sig2_v0.append(s_v0**2); sig2_b0.append(s_b0**2); sig2_b1.append(s_b1**2)

    if not sig2_v0: sys.exit("ERROR: Bootstrap analysis failed for all materials.")

    rms_v0 = math.sqrt(np.mean(sig2_v0)); rms_b0 = math.sqrt(np.mean(sig2_b0)); rms_b1 = math.sqrt(np.mean(sig2_b1))
    
    # Suivant la def. du papier (pour les poids relatifs de V0 et B0)
    w_b0_bootstrap = rms_v0 / rms_b0 if rms_b0 > 0 else 0.0
    w_b1_bootstrap = rms_b0 / rms_b1 if rms_b1 > 0 else 0.0 

    print("\n" + "-" * 50)
    print("Bootstrap-Derived Weight Calculation:")
    print(f"  => New suggested weight for B0: w_B0 = {w_b0_bootstrap:.4f}")
    print(f"  => New suggested weight for B1: w_B1 = {w_b1_bootstrap:.4f}")
    print("-" * 50)

    print("\n--- PART 2: Deriving new thresholds from nu-epsilon correlation ---")
    
    ECUT_FACTORS = [0.025, 0.05, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275, 0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.5]
    all_scan_data = {}
    for factor in ECUT_FACTORS:
        try:
            with open(input_dir / f"results_{factor}.json", 'r') as f: all_scan_data[factor] = json.load(f)
        except FileNotFoundError: print(f"Warning: Scan data for factor {factor} not found.")

    if not all_scan_data: sys.exit("ERROR: No scan data found for correlation analysis.")
        
    epsilon_values, nu_values = [], []
    for material_key in tqdm(high_ecut_curves.keys(), desc="Calculating metrics"):
        ref_bm_fit = ref_data.get('BM_fit_data', {}).get(material_key)
        ref_natoms = ref_data.get('num_atoms_in_sim_cell', {}).get(material_key)
        if not ref_bm_fit or ref_natoms is None: continue
        element, configuration = material_key.split('-')
        ref_scaling_factor = get_volume_scaling_to_formula_unit(ref_natoms, element, configuration)
        v0_ref_fu, e0_ref_fu, b0_ref, b1_ref = ref_bm_fit['min_volume']/ref_scaling_factor, ref_bm_fit['E0']/ref_scaling_factor, ref_bm_fit['bulk_modulus_ev_ang3'], ref_bm_fit['bulk_deriv']

        for factor in ECUT_FACTORS:
            scan_data = all_scan_data.get(factor)
            if not scan_data: 
                continue
            scan_bm_refined = scan_data.get('BM_fit_data', {}).get(material_key)
            scan_natoms = scan_data.get('num_atoms_in_sim_cell', {}).get(material_key)
            if not scan_bm_refined or scan_natoms is None: 
                continue
            scan_scaling_factor = get_volume_scaling_to_formula_unit(scan_natoms, element, configuration)
            v0_scan_fu, e0_scan_fu, b0_scan, b1_scan = scan_bm_refined['min_volume']/scan_scaling_factor, scan_bm_refined['E0']/scan_scaling_factor, scan_bm_refined['bulk_modulus_ev_ang3'], scan_bm_refined['bulk_deriv']

            epsilon_val = epsilon(v0_scan_fu, b0_scan, b1_scan, v0_ref_fu, b0_ref, b1_ref, 1.0, 0, 0)
            nu_val = nu(v0_scan_fu, b0_scan, b1_scan, v0_ref_fu, b0_ref, b1_ref, 1.0, w_b0_bootstrap, w_b1_bootstrap)
            epsilon_values.append(epsilon_val); nu_values.append(nu_val)

    epsilon_values, nu_values = np.array(epsilon_values), np.array(nu_values)
    mask = np.isfinite(epsilon_values) & np.isfinite(nu_values)
    slope, intercept, r_value, _, _ = linregress(epsilon_values[mask], nu_values[mask])
    
    print("\n" + "-"*50)
    print("Linear Correlation Analysis (nu vs epsilon)")
    print(f"  => Slope (C): {slope:.4f}")
    print(f"  => Intercept: {intercept:.4f}")
    print(f"  => R-squared: {r_value**2:.4f}")
    print("-" * 50)

    fig_corr, ax_corr = pl.subplots(figsize=(4, 3.5))
    ax_corr.plot(epsilon_values, nu_values, 'o', color='gray', markersize=3, alpha=0.3, label='All data points')
    x_fit = np.array([0, np.max(epsilon_values[mask])])
    ax_corr.plot(x_fit, slope * x_fit + intercept, '-', color='black', lw=1.5, label=fr'Fit: $\nu \approx {slope:.2f}\varepsilon$')
    ax_corr.set_xlabel(r'Metric $\varepsilon$ [ ]')
    ax_corr.set_ylabel(r'Metric $\nu$ [ ] (with new weights)')
    ax_corr.set_title(r'Correlation between $\nu$ and $\varepsilon$')
    ax_corr.grid(True, linestyle=':'); ax_corr.legend(loc='best')
    fig_corr.tight_layout()
    corr_plot_filename = input_dir / "nu_epsilon_correlation.pdf"
    pl.savefig(corr_plot_filename, dpi=300)
    print(f"Correlation plot saved to '{corr_plot_filename}'")
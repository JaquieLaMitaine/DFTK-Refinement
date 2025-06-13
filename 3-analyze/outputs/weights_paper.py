#!/usr/bin/env python
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pylab as pl
import tqdm

# --- Add parent directory to path to import local modules ---
# This assumes the script is run from the `3-analyze/outputs/` directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, os.pardir))
# ---

from eos_utils.eosfit_31_adapted import BM, echarge
from quantities_for_comparison import (
    birch_murnaghan, get_volume_scaling_to_formula_unit
)

# --- Configuration based on Section S3 of the paper ---
# Number of random samples to generate for each material
N_SAMPLES = 100
# Standard deviation of the Gaussian noise to add to the energy points (in eV)
NOISE_STD_DEV = 1e-5
# The set to analyze
SET_NAME = "oxides-verification-PBE-v1"
# Number of data points for the EOS curve
N_POINTS_EOS = 7
# Volume scaling range for the EOS curve (e.g., 0.94 to 1.06 for +/- 6%)
VOLUME_RANGE = np.linspace(0.94, 1.06, N_POINTS_EOS)
# ---

def calculate_weights():
    """
    Calculates the weights for the nu metric by following the procedure
    described in Section S3 of the ACWF verification paper (arXiv:2305.17274).
    """
    
    # --- Load Reference Data ---
    ref_results_file = os.path.join(SCRIPT_DIR,SET_NAME, f"dftk/results-2high_ecut.json")
    try:
        with open(ref_results_file, 'r') as f:
            ref_data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"ERROR: High-Ecut reference data file not found at: {ref_results_file}")

    all_materials = sorted(list(ref_data.get('BM_fit_data', {}).keys()))

    b0_v0_ratios = []
    b1_v0_ratios = []

    progress_bar = tqdm.tqdm(all_materials, desc="Analyzing materials")
    for material_key in progress_bar:
        # --- 1. Get Reference Parameters ---
        ref_bm_fit = ref_data.get('BM_fit_data', {}).get(material_key)
        ref_natoms = ref_data.get('num_atoms_in_sim_cell', {}).get(material_key)

        if not ref_bm_fit or ref_natoms is None:
            continue
        
        element, configuration = material_key.split('-')
        scaling_factor = get_volume_scaling_to_formula_unit(ref_natoms, element, configuration)

        v0_ref = ref_bm_fit['min_volume'] / scaling_factor
        e0_ref = ref_bm_fit['E0'] / scaling_factor
        b0_ref = ref_bm_fit['bulk_modulus_ev_ang3']
        b1_ref = ref_bm_fit['bulk_deriv']

        # --- 2. Generate "Perfect" E-V Data ---
        # Use the reference V0 as the central volume
        perfect_volumes = v0_ref * VOLUME_RANGE
        perfect_energies = birch_murnaghan(perfect_volumes, e0_ref, v0_ref, b0_ref, b1_ref)

        eta_v0_samples = []
        eta_b0_samples = []
        eta_b1_samples = []

        # --- 3-5. Add Noise, Fit, and Calculate Relative Errors ---
        for _ in range(N_SAMPLES):
            # 3. Add random noise
            noisy_energies = perfect_energies + np.random.normal(0.0, NOISE_STD_DEV, size=len(perfect_energies))
            noisy_dataset = np.array([perfect_volumes, noisy_energies]).T

            # 4. Fit the noisy data
            try:
                v0_fit, e0_fit, b0_fit_int, b1_fit, residuals = BM(noisy_dataset)
                # Convert B0 to eV/A^3 for consistency
                b0_fit = (b0_fit_int * echarge * 1.0e21) / 160.21766208
            except (ValueError, np.linalg.LinAlgError):
                # The fit can fail if the noise is too large or creates an unphysical curve
                continue
            
            # 5. Calculate relative errors eta_Y = 2 * (Y_fit - Y_ref) / (Y_fit + Y_ref)
            eta_v0 = 2 * (v0_fit - v0_ref) / (v0_fit + v0_ref)
            eta_b0 = 2 * (b0_fit - b0_ref) / (b0_fit + b0_ref)
            eta_b1 = 2 * (b1_fit - b1_ref) / (b1_fit + b1_ref)

            eta_v0_samples.append(eta_v0)
            eta_b0_samples.append(eta_b0)
            eta_b1_samples.append(eta_b1)
        
        if not eta_v0_samples:
            print(f"Warning: All fits failed for {material_key}, skipping.")
            continue
            
        # --- 6. Average the Absolute Errors (ῆ) ---
        eta_v0_avg = np.mean(np.abs(eta_v0_samples))
        eta_b0_avg = np.mean(np.abs(eta_b0_samples))
        eta_b1_avg = np.mean(np.abs(eta_b1_samples))
        
        # --- 7. Calculate Error Ratios ---
        if eta_v0_avg > 0:
            b0_v0_ratios.append(eta_b0_avg / eta_v0_avg)
            b1_v0_ratios.append(eta_b1_avg / eta_v0_avg)

    # --- 8 & 9. Generate Histograms and Determine Weights from Medians ---
    if not b0_v0_ratios or not b1_v0_ratios:
        print("ERROR: No valid error ratios were computed. Cannot determine weights.")
        return

    median_ratio_b0_v0 = np.median(b0_v0_ratios)
    median_ratio_b1_v0 = np.median(b1_v0_ratios)

    weight_b0 = 1.0 / median_ratio_b0_v0
    weight_b1 = 1.0 / median_ratio_b1_v0
    
    print("-" * 50)
    print("Derived Weights for the 'nu' Metric")
    print("-" * 50)
    print(f"Based on {len(b0_v0_ratios)} materials from the {SET_NAME} set.")
    print(f"Median of (ῆ_B₀ / ῆ_V₀): {median_ratio_b0_v0:.4f}")
    print(f"Median of (ῆ_B₁' / ῆ_V₀): {median_ratio_b1_v0:.4f}")
    print("\nCalculated weights (inverse of median ratios):")
    print(f"  w_B₀ = {weight_b0:.4f}")
    print(f"  w_B₁' = {weight_b1:.4f}")
    print("\nThe paper uses rounded values of w_B₀ = 1/20 = 0.05 and w_B₁' = 1/400 = 0.0025.")
    print("-" * 50)

    # --- Create histograms for verification ---
    fig, axes = pl.subplots(2, 1, figsize=(6, 8))
    fig.suptitle(f"Error Propagation Ratios for {SET_NAME}\n(Noise std dev: {NOISE_STD_DEV} eV, N_samples: {N_SAMPLES})", fontsize=12)

    axes[0].hist(b0_v0_ratios, bins=50, range=(0, 100))
    axes[0].axvline(median_ratio_b0_v0, color='r', linestyle='--', label=f'Median = {median_ratio_b0_v0:.2f}')
    axes[0].set_xlabel(r"$\tilde{\eta}_{B_0} / \tilde{\eta}_{V_0}$")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].set_title(r"Distribution of $B_0$ vs $V_0$ error ratios")

    axes[1].hist(b1_v0_ratios, bins=50, range=(0, 1000))
    axes[1].axvline(median_ratio_b1_v0, color='r', linestyle='--', label=f'Median = {median_ratio_b1_v0:.2f}')
    axes[1].set_xlabel(r"$\tilde{\eta}_{B'_0} / \tilde{\eta}_{V_0}$")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].set_title(r"Distribution of $B'_0$ vs $V_0$ error ratios")
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = os.path.join(SCRIPT_DIR, "weights_derivation_histogram.pdf")
    pl.savefig(plot_filename)
    print(f"\nVerification histogram plot saved to '{plot_filename}'")
    pl.close(fig)

if __name__ == "__main__":
    calculate_weights()
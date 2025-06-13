#!/usr/bin/env python
import json
import os
import sys
import numpy as np
import pylab as pl
import tqdm

from quantities_for_comparison import (
    birch_murnaghan, get_volume_scaling_to_formula_unit
)

# --- Matplotlib and LaTeX Font Configuration ---
pl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})
ECUT_FACTOR_TO_PLOT = 0.4

if __name__ == "__main__":
    try:
        SET_NAME = sys.argv[1]
    except IndexError:
        sys.exit(f"Usage: {sys.argv[0]} SET_NAME (e.g., oxides-verification-PBE-v1)")

    PLUGIN_NAME = 'dftk'

    input_dir = f"{SET_NAME}-{PLUGIN_NAME}"
    output_dir = os.path.join(input_dir, f"eos_comparison_plots_factor_{ECUT_FACTOR_TO_PLOT}")
    os.makedirs(output_dir, exist_ok=True)
    
    ref_results_file = os.path.join(input_dir, "results_2high_ecut.json")
    scan_results_file = os.path.join(input_dir, f"results_{ECUT_FACTOR_TO_PLOT}.json")

    try:
        with open(ref_results_file, 'r') as f: ref_data = json.load(f)
    except FileNotFoundError: sys.exit(f"ERROR: High-Ecut reference data file not found at: {ref_results_file}")
    
    try:
        with open(scan_results_file, 'r') as f: scan_data = json.load(f)
    except FileNotFoundError: sys.exit(f"ERROR: Scan data for factor {ECUT_FACTOR_TO_PLOT} not found at: {scan_results_file}")

    all_materials = sorted(list(ref_data.get('BM_fit_data', {}).keys()))

    progress_bar = tqdm.tqdm(all_materials, desc="Plotting EOS curves")
    for material_key in progress_bar:
        ref_bm_fit = ref_data.get('BM_fit_data', {}).get(material_key)
        ref_natoms = ref_data.get('num_atoms_in_sim_cell', {}).get(material_key)
        scan_bm_preref = scan_data.get('BM_fit_data_pre_refinement', {}).get(material_key)
        scan_bm_refined = scan_data.get('BM_fit_data', {}).get(material_key)
        scan_eos_points = scan_data.get('eos_data', {}).get(material_key)
        scan_natoms = scan_data.get('num_atoms_in_sim_cell', {}).get(material_key)

        if not all([ref_bm_fit, ref_natoms, scan_bm_preref, scan_bm_refined, scan_eos_points, scan_natoms]):
            continue

        element, configuration = material_key.split('-')
        
        ref_scaling_factor = get_volume_scaling_to_formula_unit(ref_natoms, element, configuration)
        scan_scaling_factor = get_volume_scaling_to_formula_unit(scan_natoms, element, configuration)

        v0_ref = ref_bm_fit['min_volume'] / ref_scaling_factor
        e0_ref = ref_bm_fit['E0'] / ref_scaling_factor
        b0_ref = ref_bm_fit['bulk_modulus_ev_ang3']
        b1_ref = ref_bm_fit['bulk_deriv']
        
        v0_pre = scan_bm_preref['min_volume'] / scan_scaling_factor
        e0_pre = scan_bm_preref['E0'] / scan_scaling_factor
        b0_pre = scan_bm_preref['bulk_modulus_ev_ang3']
        b1_pre = scan_bm_preref['bulk_deriv']

        v0_refi = scan_bm_refined['min_volume'] / scan_scaling_factor
        e0_refi = scan_bm_refined['E0'] / scan_scaling_factor
        b0_refi = scan_bm_refined['bulk_modulus_ev_ang3']
        b1_refi = scan_bm_refined['bulk_deriv']
        
        fig, ax = pl.subplots(figsize=(3.3, 3.0))

        vol_range = np.linspace(v0_ref * 0.92, v0_ref * 1.08, 200)

        eos_ref = birch_murnaghan(vol_range, E0=e0_ref, V0=v0_ref, B0=b0_ref, B01=b1_ref)
        ax.plot(vol_range, eos_ref - e0_ref, color='black', linestyle='--', lw=1.5, label='Reference')

        eos_pre = birch_murnaghan(vol_range, E0=e0_pre, V0=v0_pre, B0=b0_pre, B01=b1_pre)
        ax.plot(vol_range, eos_pre - e0_pre, color='gray', linestyle='--', lw=1.2, label='Pre-refinement')

        eos_refi = birch_murnaghan(vol_range, E0=e0_refi, V0=v0_refi, B0=b0_refi, B01=b1_refi)
        ax.plot(vol_range, eos_refi - e0_refi, color='black', linestyle='-', lw=1.2, label='Refined')
        
        ax.set_xlabel(r'$V$ [\AA³/atom]')
        ax.set_ylabel(r'$E - E_0$ [eV/atom]')
        ax.set_title(f'EOS Comparison for {material_key}')
        ax.grid(True, linestyle=':')
        ax.legend(loc='best')
        ax.autoscale(tight=True)
        
        fig.tight_layout()
        plot_filename = os.path.join(output_dir, f"eos_comparison_{material_key}.pdf")
        pl.savefig(plot_filename, dpi=300)
        pl.close(fig)

    print(f"\nEOS comparison plots generated in '{output_dir}'.")
#!/usr/bin/env python
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pylab as pl
import tqdm

from quantities_for_comparison import get_volume_scaling_to_formula_unit

# --- Config ---
ECUT_FACTORS = [0.025, 0.05, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275, 0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.5]
# ---

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
# ---

if __name__ == "__main__":
    try:
        SET_NAME = sys.argv[1]
    except IndexError:
        sys.exit(f"Usage: {sys.argv[0]} SET_NAME (e.g., oxides-verification-PBE-v1)")

    PLUGIN_NAME = 'dftk'

    # --- Define file paths ---
    input_dir = f"{SET_NAME}-{PLUGIN_NAME}"
    output_dir = os.path.join(input_dir, "parameter_convergence_plots") 
    os.makedirs(output_dir, exist_ok=True)
    
    ref_results_file = os.path.join(input_dir, "results_2high_ecut.json")

    # --- Load Data ---
    try:
        with open(ref_results_file, 'r') as f:
            ref_data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"ERROR: High-Ecut reference data file not found at: {ref_results_file}")

    all_scan_data = {}
    for factor in ECUT_FACTORS:
        scan_results_file = os.path.join(input_dir, f"results_{factor}.json")
        try:
            with open(scan_results_file, 'r') as f:
                all_scan_data[factor] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Scan data file not found for factor {factor}, skipping: {scan_results_file}")
            continue

    summary_data = {
        'v0_diff_refined': defaultdict(list), 'v0_diff_preref': defaultdict(list),
        'e0_diff_refined': defaultdict(list), 'e0_diff_preref': defaultdict(list),
        'b0_diff_refined': defaultdict(list), 'b0_diff_preref': defaultdict(list),
        'b1_diff_refined': defaultdict(list), 'b1_diff_preref': defaultdict(list),
    }


    # --- Main processing loop ---
    all_materials_in_ref = sorted(list(ref_data.get('BM_fit_data', {}).keys()))

    progress_bar = tqdm.tqdm(all_materials_in_ref, desc="Processing materials")
    for material_key in progress_bar:
        progress_bar.set_description(f"{material_key:12s}")

        ref_bm_fit = ref_data.get('BM_fit_data', {}).get(material_key)
        ref_natoms = ref_data.get('num_atoms_in_sim_cell', {}).get(material_key)
        if not ref_bm_fit or ref_natoms is None:
            print(f"Skipping {material_key}: missing valid reference data.")
            continue

        element, configuration = material_key.split('-')
        ref_scaling_factor = get_volume_scaling_to_formula_unit(ref_natoms, element, configuration)
        v0_ref_fu = ref_bm_fit['min_volume'] / ref_scaling_factor
        e0_ref_fu = ref_bm_fit['E0'] / ref_scaling_factor
        b0_ref = ref_bm_fit['bulk_modulus_ev_ang3']
        b1_ref = ref_bm_fit['bulk_deriv']

        ecuts_ha = []
        v0_diff_refined, v0_diff_preref = [], []
        e0_diff_refined, e0_diff_preref = [], []
        b0_diff_refined, b0_diff_preref = [], []
        b1_diff_refined, b1_diff_preref = [], []

        for factor in ECUT_FACTORS:
            scan_data = all_scan_data.get(factor)
            if not scan_data:
                continue
                
            ecut_ha = scan_data.get('ecuts_used_Ha', {}).get(material_key)
            if ecut_ha is None:
                continue
            
            scan_natoms = scan_data.get('num_atoms_in_sim_cell', {}).get(material_key)
            if scan_natoms is None: continue

            scan_scaling_factor = get_volume_scaling_to_formula_unit(scan_natoms, element, configuration)

            scan_bm_fit = scan_data.get('BM_fit_data', {}).get(material_key)
            if scan_bm_fit:
                v0_scan_fu = scan_bm_fit['min_volume'] / scan_scaling_factor
                e0_scan_fu = scan_bm_fit['E0'] / scan_scaling_factor
                b0_scan = scan_bm_fit['bulk_modulus_ev_ang3']
                b1_scan = scan_bm_fit['bulk_deriv']
                
                v0_diff_refined.append(abs(v0_scan_fu - v0_ref_fu))
                e0_diff_refined.append(abs(e0_scan_fu - e0_ref_fu))
                b0_diff_refined.append(abs(b0_scan - b0_ref))
                b1_diff_refined.append(abs(b1_scan - b1_ref))
                summary_data['v0_diff_refined'][factor].append(abs(v0_scan_fu - v0_ref_fu))
                summary_data['e0_diff_refined'][factor].append(abs(e0_scan_fu - e0_ref_fu))
                summary_data['b0_diff_refined'][factor].append(abs(b0_scan - b0_ref))
                summary_data['b1_diff_refined'][factor].append(abs(b1_scan - b1_ref))
            else: 
                v0_diff_refined.append(np.nan)
                e0_diff_refined.append(np.nan)
                b0_diff_refined.append(np.nan)
                b1_diff_refined.append(np.nan)

            scan_bm_fit_preref = scan_data.get('BM_fit_data_pre_refinement', {}).get(material_key)
            if scan_bm_fit_preref:
                v0_scan_pre_fu = scan_bm_fit_preref['min_volume'] / scan_scaling_factor
                e0_scan_pre_fu = scan_bm_fit_preref['E0'] / scan_scaling_factor
                b0_scan_pre = scan_bm_fit_preref['bulk_modulus_ev_ang3']
                b1_scan_pre = scan_bm_fit_preref['bulk_deriv']

                v0_diff_preref.append(abs(v0_scan_pre_fu - v0_ref_fu))
                e0_diff_preref.append(abs(e0_scan_pre_fu - e0_ref_fu))
                b0_diff_preref.append(abs(b0_scan_pre - b0_ref))
                b1_diff_preref.append(abs(b1_scan_pre - b1_ref))
                
                summary_data['v0_diff_preref'][factor].append(abs(v0_scan_pre_fu - v0_ref_fu))
                summary_data['e0_diff_preref'][factor].append(abs(e0_scan_pre_fu - e0_ref_fu))
                summary_data['b0_diff_preref'][factor].append(abs(b0_scan_pre - b0_ref))
                summary_data['b1_diff_preref'][factor].append(abs(b1_scan_pre - b1_ref))
            else:
                v0_diff_preref.append(np.nan)
                e0_diff_preref.append(np.nan)
                b0_diff_preref.append(np.nan)
                b1_diff_preref.append(np.nan)
            
            ecuts_ha.append(ecut_ha)

        if not ecuts_ha:
            print(f"No valid scan data found for {material_key}. Skipping plots.")
            continue

        # --- Plotting V0 Convergence ---
        fig_v0, ax_v0 = pl.subplots(figsize=(3.3, 3.0))
        ax_v0.plot(ecuts_ha, v0_diff_preref, 's--',mfc='none', color='black', label='Pre-refinement', markersize=3, linewidth=0.8)
        ax_v0.plot(ecuts_ha, v0_diff_refined, 's-', color='black', label='Refined', markersize=3, linewidth=1.0)
        ax_v0.set_yscale('log')
        ax_v0.set_xlabel(r'$E_{\mathrm{cut}}$ [Ha]')
        ax_v0.set_ylabel(r'$|V_0 - V_{0, \mathrm{ref}}|$ [\AA³/atom]')
        ax_v0.grid(True, linestyle=':')
        ax_v0.legend(loc='upper right')
        ax_v0.set_title(f"$V_0$ Convergence for {material_key}", pad=8)
        fig_v0.tight_layout()
        plot_filename_v0 = os.path.join(output_dir, f"{material_key}_V0_convergence.pdf")
        pl.savefig(plot_filename_v0, dpi=300)
        pl.close(fig_v0)

        # --- Plotting E0 Convergence ---
        fig_e0, ax_e0 = pl.subplots(figsize=(3.3, 3.0))
        ax_e0.plot(ecuts_ha, e0_diff_preref, 's--',mfc='none', color='black', label='Pre-refinement', markersize=3, linewidth=0.8)
        ax_e0.plot(ecuts_ha, e0_diff_refined,   's-',  color='black', label='Refined', markersize=3, linewidth=1.0)
        ax_e0.set_yscale('log')
        ax_e0.set_xlabel(r'$E_{\mathrm{cut}}$ [Ha]')
        ax_e0.set_ylabel(r'$|E_0 - E_{0, \mathrm{ref}}|$ [eV/atom]')
        ax_e0.grid(True, linestyle=':')
        ax_e0.legend(loc='upper right')
        ax_e0.set_title(f"$E_0$ convergence for {material_key}", pad=8)
        fig_e0.tight_layout()
        plot_filename_e0 = os.path.join(output_dir, f"{material_key}_E0_convergence.pdf")
        pl.savefig(plot_filename_e0, dpi=300)
        pl.close(fig_e0)

        # --- Plotting B0 Convergence ---
        fig_b0, ax_b0 = pl.subplots(figsize=(3.3, 3.0))
        ax_b0.plot(ecuts_ha, b0_diff_preref, 's--',mfc='none', color='black', label='Pre-refinement', markersize=3, linewidth=0.8)
        ax_b0.plot(ecuts_ha, b0_diff_refined,   's-',  color='black', label='Refined', markersize=3, linewidth=1.0)
        ax_b0.set_yscale('log')
        ax_b0.set_xlabel(r'$E_{\mathrm{cut}}$ [Ha]')
        ax_b0.set_ylabel(r'$|B_0 - B_{0, \mathrm{ref}}|$ [eV/\AA³]')
        ax_b0.grid(True, linestyle=':')
        ax_b0.legend(loc='upper right')
        ax_b0.set_title(f"$B_0$ Convergence for {material_key}", pad=8)
        fig_b0.tight_layout()
        plot_filename_b0 = os.path.join(output_dir, f"{material_key}_B0_convergence.pdf")
        pl.savefig(plot_filename_b0, dpi=300)
        pl.close(fig_b0)

        # --- Plotting B1 Convergence ---
        fig, ax = pl.subplots(figsize=(3.3, 3.0))
        ax.plot(ecuts_ha, b1_diff_preref, 's--',mfc='none', label='Pre-refinement', color='black', markersize=3, linewidth=0.8)
        ax.plot(ecuts_ha, b1_diff_refined, 's-', label='Refined', color='black', markersize=3, linewidth=1.0)
        ax.set_yscale('log')
        ax.set_xlabel(r'$E_{\mathrm{cut}}$ [Ha]')
        ax.set_ylabel(r"$|B'_0 - B'_{0, \mathrm{ref}}|$ [ ]")
        ax.grid(True, linestyle=':')
        ax.legend(loc='upper right')
        ax.set_title(f"$B_0'$ Convergence for {material_key}", pad=8)
        fig.tight_layout()
        plot_filename_b1 = os.path.join(output_dir, f"{material_key}_B1_convergence.pdf")
        pl.savefig(plot_filename_b1, dpi=300)
        pl.close(fig)

    print(f"\nIndividual parameter convergence plots successfully generated in '{output_dir}'.")
    
    print("\nGenerating summary box-plots (grouped by Ecut factor)...")
    
    ecut_factors_sorted = sorted(summary_data['e0_diff_refined'].keys())
    ecut_labels = [f"{factor:.3f}" for factor in ecut_factors_sorted]
    
    def create_summary_boxplot(param_name, y_label, title_suffix):
        fig, ax = pl.subplots(figsize=(3.3, 3))
        
        preref_data = [summary_data[f'{param_name}_diff_preref'][key] for key in ecut_factors_sorted]
        refined_data = [summary_data[f'{param_name}_diff_refined'][key] for key in ecut_factors_sorted]
        
        positions = np.arange(len(ecut_factors_sorted))
        
        boxprops_preref = {'facecolor': 'none', 'edgecolor': 'gray'}
        medianprops_preref = {'color': 'gray'}
        boxprops_refined = {'facecolor': 'none', 'edgecolor': 'black'}
        medianprops_refined = {'color': 'black'}

        bp1 = ax.boxplot(preref_data, positions=positions, widths=0.35, patch_artist=True,
                         boxprops=boxprops_preref, medianprops=medianprops_preref,
                         whiskerprops=dict(color='gray'), capprops=dict(color='gray'), showfliers=False)

        bp2 = ax.boxplot(refined_data, positions=positions, widths=0.35, patch_artist=True,
                         boxprops=boxprops_refined, medianprops=medianprops_refined,
                         whiskerprops=dict(color='black'), capprops=dict(color='black'), showfliers=False)

        ax.set_yscale('log')
        ax.set_ylabel(y_label)
        ax.set_xlabel(r'Ecut Factor') 
        ax.set_title(f'Distribution of {title_suffix}', pad=8)
        num_ticks_to_show = 5
        total_ticks = len(ecut_factors_sorted)
        tick_spacing = max(1, total_ticks // num_ticks_to_show)

        display_positions = positions[::tick_spacing]
        display_labels = [ecut_labels[i] for i in display_positions]
        
        ax.set_xticks(display_positions)
        ax.set_xticklabels(display_labels, rotation=0)
        
        ax.grid(True, axis='y')
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', lw=2, label='Pre-refinement'),
            Line2D([0], [0], color='black', lw=2, label='Refined')
        ]
        ax.legend(handles=legend_elements, loc='lower left')
        
        fig.tight_layout()
        plot_filename = os.path.join(output_dir, f"SUMMARY_boxplot_vs_factor_{param_name}_error.pdf")
        pl.savefig(plot_filename, dpi=300)
        pl.close(fig)
        print(f"Summary plot saved to '{plot_filename}'.")

    create_summary_boxplot('v0', r'$|V_0 - V_{0, \mathrm{ref}}|$ [\AA³/atom]', '$V_0$')
    create_summary_boxplot('e0', r'$|E_0 - E_{0, \mathrm{ref}}|$ [eV/atom]', '$E_0$')
    create_summary_boxplot('b0', r'$|B_0 - B_{0, \mathrm{ref}}|$ [eV/\AA³]', '$B_0$')
    create_summary_boxplot('b1', r"$|B'_0 - B'_{0, \mathrm{ref}}|$", "$B_0'$")
    
    print("\nAll summary plots generated.")

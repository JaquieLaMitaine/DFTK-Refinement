#!/usr/bin/env python
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pylab as pl
import tqdm
from matplotlib.ticker import NullLocator

from quantities_for_comparison import (
    delta, nu, epsilon, get_volume_scaling_to_formula_unit
)

# --- Config ---
ECUT_FACTORS = [0.025, 0.05, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275, 0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.5]
NU_PRIME_WEIGHT_B0 = 0.0568
NU_PRIME_WEIGHT_B1 = 0.2755
PREFACTOR = 1.0
THRESHOLDS = {
    'delta': {'excellent': 0.3, 'good': 0.95},
    'nu': {'excellent': 0.043, 'good': 0.14},
    'epsilon': {'excellent': 0.06, 'good': 0.20},
}
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

def find_convergence_ecut(ecuts, values, threshold):
    last_bad_index = -1
    for i, value in enumerate(values):
        if not np.isnan(value) and value > threshold:
            last_bad_index = i
    if last_bad_index == len(ecuts) - 1:
        return np.nan
    else:
        return ecuts[last_bad_index + 1]

if __name__ == "__main__":
    try:
        SET_NAME = sys.argv[1]
    except IndexError:
        sys.exit(f"Usage: {sys.argv[0]} SET_NAME (e.g., oxides-verification-PBE-v1)")

    PLUGIN_NAME = 'dftk'
    input_dir = f"{SET_NAME}-{PLUGIN_NAME}"
    output_dir_individual = os.path.join(input_dir, "convergence_plots_comparison_individual")
    output_dir_summary = os.path.join(input_dir, "convergence_plots_comparison_summary")
    os.makedirs(output_dir_individual, exist_ok=True)
    os.makedirs(output_dir_summary, exist_ok=True)
    
    ref_results_file = os.path.join(input_dir, "results_2high_ecut.json")

    try:
        with open(ref_results_file, 'r') as f: ref_data = json.load(f)
    except FileNotFoundError: sys.exit(f"ERROR: High-Ecut reference data file not found: {ref_results_file}")
    all_scan_data = {}
    for factor in ECUT_FACTORS:
        try:
            with open(os.path.join(input_dir, f"results_{factor}.json"), 'r') as f: all_scan_data[factor] = json.load(f)
        except FileNotFoundError: print(f"Warning: Scan data for factor {factor} not found, skipping.")
    summary_convergence_data = {'material_keys': [], 'delta_good': [], 'delta_excellent': [], 'nu_good': [], 'nu_excellent': [], 'epsilon_good': [], 'epsilon_excellent': []}
    summary_metrics_data = {'delta_refined': defaultdict(list), 'delta_preref': defaultdict(list), 'nu_refined': defaultdict(list), 'nu_preref': defaultdict(list), 'epsilon_refined': defaultdict(list), 'epsilon_preref': defaultdict(list)}
    all_materials_in_ref = sorted(list(ref_data.get('BM_fit_data', {}).keys()))
    for material_key in tqdm.tqdm(all_materials_in_ref, desc="Processing materials"):
        ref_bm_fit = ref_data.get('BM_fit_data', {}).get(material_key); ref_natoms = ref_data.get('num_atoms_in_sim_cell', {}).get(material_key)
        if not ref_bm_fit or ref_natoms is None: continue
        element, configuration = material_key.split('-'); ref_scaling_factor = get_volume_scaling_to_formula_unit(ref_natoms, element, configuration)
        v0_ref, e0_ref, b0_ref, b1_ref = ref_bm_fit['min_volume']/ref_scaling_factor, ref_bm_fit['E0']/ref_scaling_factor, ref_bm_fit['bulk_modulus_ev_ang3'], ref_bm_fit['bulk_deriv']
        ecuts_ha = []; metrics_refined = {'delta': [], 'nu': [], 'epsilon': []}; metrics_preref = {'delta': [], 'nu': [], 'epsilon': []}
        for factor in ECUT_FACTORS:
            scan_data = all_scan_data.get(factor);
            if not scan_data: continue
            ecut_ha = scan_data.get('ecuts_used_Ha', {}).get(material_key);
            if ecut_ha is None: continue
            scan_natoms = scan_data.get('num_atoms_in_sim_cell', {}).get(material_key);
            if scan_natoms is None: continue
            scan_scaling_factor = get_volume_scaling_to_formula_unit(scan_natoms, element, configuration)
            scan_bm_fit = scan_data.get('BM_fit_data', {}).get(material_key)
            if scan_bm_fit:
                v0_scan, e0_scan, b0_scan, b1_scan = scan_bm_fit['min_volume']/scan_scaling_factor, scan_bm_fit['E0']/scan_scaling_factor, scan_bm_fit['bulk_modulus_ev_ang3'], scan_bm_fit['bulk_deriv']
                delta_val, nu_val, eps_val = delta(v0_scan, b0_scan, b1_scan, v0_ref, b0_ref, b1_ref, 1.0, 0, 0), nu(v0_scan, b0_scan, b1_scan, v0_ref, b0_ref, b1_ref, PREFACTOR, NU_PRIME_WEIGHT_B0, NU_PRIME_WEIGHT_B1), epsilon(v0_scan, b0_scan, b1_scan, v0_ref, b0_ref, b1_ref, PREFACTOR, 0, 0)
                metrics_refined['delta'].append(delta_val); metrics_refined['nu'].append(nu_val); metrics_refined['epsilon'].append(eps_val)
                summary_metrics_data['delta_refined'][factor].append(delta_val); summary_metrics_data['nu_refined'][factor].append(nu_val); summary_metrics_data['epsilon_refined'][factor].append(eps_val)
            else: metrics_refined['delta'].append(np.nan); metrics_refined['nu'].append(np.nan); metrics_refined['epsilon'].append(np.nan)
            scan_bm_fit_preref = scan_data.get('BM_fit_data_pre_refinement', {}).get(material_key)
            if scan_bm_fit_preref:
                v0_scan_pre, e0_scan_pre, b0_scan_pre, b1_scan_pre = scan_bm_fit_preref['min_volume']/scan_scaling_factor, scan_bm_fit_preref['E0']/scan_scaling_factor, scan_bm_fit_preref['bulk_modulus_ev_ang3'], scan_bm_fit_preref['bulk_deriv']
                delta_val_pre, nu_val_pre, eps_val_pre = delta(v0_scan_pre, b0_scan_pre, b1_scan_pre, v0_ref, b0_ref, b1_ref, 1.0, 0, 0), nu(v0_scan_pre, b0_scan_pre, b1_scan_pre, v0_ref, b0_ref, b1_ref, PREFACTOR, NU_PRIME_WEIGHT_B0, NU_PRIME_WEIGHT_B1), epsilon(v0_scan_pre, b0_scan_pre, b1_scan_pre, v0_ref, b0_ref, b1_ref, PREFACTOR, 0, 0)
                metrics_preref['delta'].append(delta_val_pre); metrics_preref['nu'].append(nu_val_pre); metrics_preref['epsilon'].append(eps_val_pre)
                summary_metrics_data['delta_preref'][factor].append(delta_val_pre); summary_metrics_data['nu_preref'][factor].append(nu_val_pre); summary_metrics_data['epsilon_preref'][factor].append(eps_val_pre)
            else: metrics_preref['delta'].append(np.nan); metrics_preref['nu'].append(np.nan); metrics_preref['epsilon'].append(np.nan)
            ecuts_ha.append(ecut_ha)
        if not ecuts_ha: continue

        # --- Plot Delta ---
        fig_delta, ax_delta = pl.subplots(figsize=(3.3, 3.0))
        ax_delta.set_title(f'$\Delta$ Convergence for {material_key}',pad=8)

        ax_delta.plot(ecuts_ha, metrics_preref['delta'], 's--', mfc='none', color='black', label='Pre-refinement', markersize=3, linewidth=0.8)
        ax_delta.plot(ecuts_ha, metrics_refined['delta'], 's-', color='black', label='Refined', markersize=3, linewidth=1.0)

        ax_delta.set_yscale('log')
        ax_delta.set_ylabel(r'$\Delta$ [meV/atom]')
        ax_delta.set_xlabel(r'$E_{\mathrm{cut}}$ [Ha]')
        ax_delta.grid(True, linestyle=':')

        ax_delta.axhline(THRESHOLDS['delta']['good'], linestyle='dotted', color='black', label='good')
        ax_delta.axhline(THRESHOLDS['delta']['excellent'], linestyle='--', color='black', label='excellent')

        ax_delta.legend(loc='best')
        fig_delta.tight_layout()
        plot_filename_delta = os.path.join(output_dir_individual, f"{material_key}_delta_convergence.pdf")
        pl.savefig(plot_filename_delta, dpi=300)
        pl.close(fig_delta)


        # --- Plot Nu ---
        fig_nu, ax_nu = pl.subplots(figsize=(3.3, 3.0))
        ax_nu.set_title(f'$\\nu$ Convergence for {material_key}',pad = 8)

        ax_nu.plot(ecuts_ha, metrics_preref['nu'], 's--', mfc='none', color='black', label='Pre-refinement', markersize=3, linewidth=0.8)
        ax_nu.plot(ecuts_ha, metrics_refined['nu'], 's-', color='black', label='Refined', markersize=3, linewidth=1.0)

        ax_nu.set_yscale('log')
        ax_nu.set_ylabel(r"$\nu$ [ ]")
        ax_nu.set_xlabel(r'$E_{\mathrm{cut}}$ [Ha]')
        ax_nu.grid(True, linestyle=':')
        

        ax_nu.axhline(THRESHOLDS['nu']['good'], linestyle='dotted', color='black', label='good')
        ax_nu.axhline(THRESHOLDS['nu']['excellent'], linestyle='--', color='black', label='excellent')
        ax_nu.legend(loc='best')
        fig_nu.tight_layout()
        plot_filename_nu = os.path.join(output_dir_individual, f"{material_key}_nu_convergence.pdf")
        pl.savefig(plot_filename_nu, dpi=300)
        pl.close(fig_nu)


        # --- Plot Epsilon ---
        fig_eps, ax_eps = pl.subplots(figsize=(3.3, 3.0))
        ax_eps.set_title(f'$\\varepsilon$ Convergence for {material_key}',pad = 8)

        ax_eps.plot(ecuts_ha, metrics_preref['epsilon'], 's--', mfc='none', color='black', label='Pre-refinement', markersize=3, linewidth=0.8)
        ax_eps.plot(ecuts_ha, metrics_refined['epsilon'], 's-', color='black', label='Refined', markersize=3, linewidth=1.0)

        ax_eps.set_yscale('log')
        ax_eps.set_ylabel(r'$\varepsilon$ [ ]')
        ax_eps.set_xlabel(r'$E_{\mathrm{cut}}$ [Ha]')
        ax_eps.grid(True, linestyle=':')
        

        ax_eps.axhline(THRESHOLDS['epsilon']['good'], linestyle='dotted', color='black', label='good')
        ax_eps.axhline(THRESHOLDS['epsilon']['excellent'], linestyle='--', color='black', label='excellent')
        ax_eps.legend(loc='best')
        fig_eps.tight_layout()
        plot_filename_eps = os.path.join(output_dir_individual, f"{material_key}_epsilon_convergence.pdf")
        pl.savefig(plot_filename_eps, dpi=300)
        pl.close(fig_eps)

        summary_convergence_data['material_keys'].append(material_key)
        for metric in ['delta', 'nu', 'epsilon']:
            for level in ['good', 'excellent']:
                threshold_value = THRESHOLDS[metric][level]
                summary_convergence_data[f'{metric}_{level}'].append(find_convergence_ecut(ecuts_ha, metrics_refined[metric], threshold_value))


    print(f"\nIndividual convergence plots generated in '{output_dir_individual}'.")
    
    print("\nGenerating summary convergence threshold plot...")
    fig_summary, ax_summary = pl.subplots(figsize=(6.6, 3.0))
    x_pos = np.arange(len(summary_convergence_data['material_keys']))
    ax_summary.plot(x_pos, summary_convergence_data['delta_good'], '^', mfc='none', color='gray')
    ax_summary.plot(x_pos, summary_convergence_data['delta_excellent'], '^', mfc='none', color='black')
    ax_summary.plot(x_pos, summary_convergence_data['nu_good'], 's', mfc='none', color='gray')
    ax_summary.plot(x_pos, summary_convergence_data['nu_excellent'], 's', mfc='none', color='black')
    ax_summary.plot(x_pos, summary_convergence_data['epsilon_good'], 'o', mfc='none', color='gray')
    ax_summary.plot(x_pos, summary_convergence_data['epsilon_excellent'], 'o', mfc='none', color='black')
    ax_summary.set_xticks(x_pos)
    ax_summary.set_xticklabels(summary_convergence_data['material_keys'], rotation=90)
    ax_summary.set_xlim(left=-0.5, right=len(x_pos) - 0.5)
    ax_summary.set_ylabel(r'$E_{\mathrm{cut}}$ [Ha]')
    ax_summary.grid(True, axis='y', linestyle=':')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^',mfc='none', color='w', label=r'$\Delta$', markeredgecolor='black'), 
        Line2D([0], [0], marker='s',mfc='none', color='w', label=r"$\nu$", markeredgecolor='black'), 
        Line2D([0], [0], marker='o', mfc='none',color='w', label=r'$\varepsilon$', markeredgecolor='black'), 
        Line2D([0], [0], color='gray', lw=2, label='Good'), 
        Line2D([0], [0], color='black', lw=2, label='Excellent')]
    ax_summary.legend(handles=legend_elements, loc='best', ncol=1); fig_summary.tight_layout()
    summary_plot_filename = os.path.join(output_dir_summary, f"SUMMARY_convergence_thresholds.pdf")
    pl.savefig(summary_plot_filename, dpi=300); pl.close(fig_summary)
    print(f"Summary convergence plot saved to '{summary_plot_filename}'.")


    print("\nGenerating summary box-plots for metrics (grouped by Ecut factor)...")
    
    ecut_factors_sorted = sorted(summary_metrics_data['delta_refined'].keys())
    ecut_labels = [f"{factor:.3f}" for factor in ecut_factors_sorted]
    
    fig_box, (ax_delta, ax_nu, ax_eps) = pl.subplots(3, 1, figsize=(3.3, 8.0), sharex=True)

    plot_params = {
        'delta': (ax_delta, r'$\Delta$ [meV/atom]', THRESHOLDS['delta']),
        'nu': (ax_nu, r'$\nu$ [ ]', THRESHOLDS['nu']),
        'epsilon': (ax_eps, r'$\varepsilon$ [ ]', THRESHOLDS['epsilon'])
    }
    
    for metric_name, (ax, y_label, thresholds) in plot_params.items():
        preref_data = [summary_metrics_data[f'{metric_name}_preref'][factor] for factor in ecut_factors_sorted]
        refined_data = [summary_metrics_data[f'{metric_name}_refined'][factor] for factor in ecut_factors_sorted]
        positions = np.arange(len(ecut_factors_sorted))
        
        boxprops_preref = {'facecolor': 'none', 'edgecolor': 'gray'}
        medianprops_preref = {'color': 'gray'}
        boxprops_refined = {'facecolor': 'none', 'edgecolor': 'black'}
        medianprops_refined = {'color': 'black'}

        if metric_name == 'delta':
            y_ticks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1,1e2,1e3]
        elif metric_name == 'nu':
            y_ticks = [1e-8,1e-7,1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
        elif metric_name == 'epsilon':
            y_ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
        

        ax.boxplot(preref_data, positions=positions, widths=0.35, patch_artist=True,
                   boxprops=boxprops_preref, medianprops=medianprops_preref,
                   whiskerprops=dict(color='gray'), capprops=dict(color='gray'), showfliers=False)
        ax.boxplot(refined_data, positions=positions, widths=0.35, patch_artist=True,
                   boxprops=boxprops_refined, medianprops=medianprops_refined,
                   whiskerprops=dict(color='black'), capprops=dict(color='black'), showfliers=False)
        
        ax.set_yscale('log'); ax.set_ylabel(y_label)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_minor_locator(NullLocator())
        ax.axhline(thresholds['good'], linestyle='dotted', color='black')
        ax.axhline(thresholds['excellent'], linestyle='--', color='black')
        ax.grid(True, axis='y')

    ax_delta.legend([pl.Line2D([0], [0], color='gray', lw=2), 
                     pl.Line2D([0], [0], color='black', lw=2),
                     pl.Line2D([0], [0], linestyle='dotted', color='black'),
                     pl.Line2D([0], [0], linestyle='--', color='black')],
                     ['Pre-refinement', 'Refined','Good','Excellent'], 
                     loc='best',
                     
                     )
    ax_eps.set_xlabel(r'Ecut Factor')
    
    num_ticks_to_show = 5; total_ticks = len(ecut_factors_sorted)
    tick_spacing = max(1, total_ticks // num_ticks_to_show)
    display_positions = positions[::tick_spacing]
    display_labels = [ecut_labels[i] for i in display_positions]
    pl.setp(ax_eps, xticks=display_positions, xticklabels=display_labels)
    
    fig_box.tight_layout()
    plot_filename_box = os.path.join(output_dir_summary, f"SUMMARY_boxplot_vs_factor_metrics.pdf")
    pl.savefig(plot_filename_box, dpi=300)
    pl.close(fig_box)
    print(f"Summary boxplot for all metrics saved to '{plot_filename_box}'.")
#!/usr/bin/env python
import json
import os
import sys
from collections import defaultdict
import numpy as np
import pylab as pl
import tqdm

from quantities_for_comparison import (
    delta, nu, epsilon, get_volume_scaling_to_formula_unit
)

# --- Matplotlib and LaTeX Font Configuration ---
pl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 8,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})
def split_violin_plot(ax, data1, data2, positions, width=0.8):
    for pos, d1, d2 in zip(positions, data1, data2):
        if len(d1) > 1:
            parts1 = ax.violinplot(d1, [pos], showmeans=False, showmedians=False, showextrema=False, widths=width)
            for pc in parts1['bodies']:
                pc.set_facecolor('gray')
                pc.set_edgecolor('gray')
                pc.set_alpha(0.6)
                m = np.mean(pc.get_paths()[0].vertices[:, 0])
                pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], -np.inf, m)
        
        if len(d2) > 1:
            parts2 = ax.violinplot(d2, [pos], showmeans=False, showmedians=False, showextrema=False, widths=width)
            for pc in parts2['bodies']:
                pc.set_facecolor('black')
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)
                m = np.mean(pc.get_paths()[0].vertices[:, 0])
                pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf)

    for pos, d1, d2 in zip(positions, data1, data2):
        if len(d1) > 0:
            q1, med, q3 = np.percentile(d1, [25, 50, 75])
            ax.plot([pos - width*0.05, pos - width*0.2], [med, med], color='white', lw=1.2)
            ax.plot([pos - width*0.125], [q1], marker='_', color='white', ms=5)
            ax.plot([pos - width*0.125], [q3], marker='_', color='white', ms=5)

        if len(d2) > 0:
            q1, med, q3 = np.percentile(d2, [25, 50, 75])
            ax.plot([pos + width*0.05, pos + width*0.2], [med, med], color='white', lw=1.2)
            ax.plot([pos + width*0.125], [q1], marker='_', color='white', ms=5)
            ax.plot([pos + width*0.125], [q3], marker='_', color='white', ms=5)

if __name__ == "__main__":
    try:
        SET_NAME = sys.argv[1]
    except IndexError: sys.exit(f"Usage: {sys.argv[0]} SET_NAME (e.g., oxides-verification-PBE-v1)")

    PLUGIN_NAME = 'dftk'

    NU_PRIME_WEIGHT_B0 = 5.7381; NU_PRIME_WEIGHT_B1 = 1.5522; PREFACTOR = 1.0
    THRESHOLDS = {'delta': {'excellent': 0.3, 'good': 0.95}, 'nu': {'excellent': 0.10, 'good': 0.33}, 'epsilon': {'excellent': 0.06, 'good': 0.20}}
    ECUT_FACTORS = [0.025, 0.05, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275, 0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.5]

    input_dir = f"{SET_NAME}-{PLUGIN_NAME}"
    output_dir_individual = os.path.join(input_dir, "timing_vs_metrics_plots")
    output_dir_summary = os.path.join(input_dir, "timing_vs_metrics_summary") # Dossier pour tous les plots de synthèse
    os.makedirs(output_dir_individual, exist_ok=True)
    os.makedirs(output_dir_summary, exist_ok=True)
    
    ref_results_file = os.path.join(input_dir, "results_2high_ecut.json")

    try:
        with open(ref_results_file, 'r') as f: ref_data = json.load(f)
    except FileNotFoundError: sys.exit(f"ERROR: High-Ecut reference data file not found at: {ref_results_file}")
    all_scan_data = {}
    for factor in ECUT_FACTORS:
        try:
            with open(os.path.join(input_dir, f"results_{factor}.json"), 'r') as f: all_scan_data[factor] = json.load(f)
        except FileNotFoundError: print(f"Warning: Scan data for factor {factor} not found.")

    summary_data = {m: {'preref': [], 'refined': []} for m in ['delta', 'nu', 'epsilon']}
    all_materials_in_ref = sorted(list(ref_data.get('BM_fit_data', {}).keys()))

    for material_key in tqdm.tqdm(all_materials_in_ref, desc="Processing materials"):
        ref_bm_fit = ref_data.get('BM_fit_data', {}).get(material_key); ref_natoms = ref_data.get('num_atoms_in_sim_cell', {}).get(material_key)
        if not ref_bm_fit or ref_natoms is None: continue
        element, configuration = material_key.split('-'); ref_scaling_factor = get_volume_scaling_to_formula_unit(ref_natoms, element, configuration)
        v0_ref_fu, e0_ref_fu, b0_ref, b1_ref = ref_bm_fit['min_volume']/ref_scaling_factor, ref_bm_fit['E0']/ref_scaling_factor, ref_bm_fit['bulk_modulus_ev_ang3'], ref_bm_fit['bulk_deriv']
        for factor in ECUT_FACTORS:
            scan_data = all_scan_data.get(factor);
            if not scan_data: continue
            scan_natoms = scan_data.get('num_atoms_in_sim_cell', {}).get(material_key);
            if scan_natoms is None: continue
            scan_scaling_factor = get_volume_scaling_to_formula_unit(scan_natoms, element, configuration)
            scan_bm_preref = scan_data.get('BM_fit_data_pre_refinement', {}).get(material_key); timings_preref = scan_data.get('timings_scf_s', {}).get(material_key)
            if scan_bm_preref and timings_preref and all(t is not None for t in timings_preref):
                t_tot = sum(timings_preref) / 1000.0; v0_pre, e0_pre, b0_pre, b1_pre = (scan_bm_preref['min_volume']/scan_scaling_factor, scan_bm_preref['E0']/scan_scaling_factor, scan_bm_preref['bulk_modulus_ev_ang3'], scan_bm_preref['bulk_deriv'])
                summary_data['delta']['preref'].append((t_tot, delta(v0_pre, b0_pre, b1_pre, v0_ref_fu, b0_ref, b1_ref, 1.0, 0, 0)))
                summary_data['nu']['preref'].append((t_tot, nu(v0_pre, b0_pre, b1_pre, v0_ref_fu, b0_ref, b1_ref, PREFACTOR, NU_PRIME_WEIGHT_B0, NU_PRIME_WEIGHT_B1)))
                summary_data['epsilon']['preref'].append((t_tot, epsilon(v0_pre, b0_pre, b1_pre, v0_ref_fu, b0_ref, b1_ref, PREFACTOR, 0, 0)))
            scan_bm_refined = scan_data.get('BM_fit_data', {}).get(material_key); timings_refined = scan_data.get('timings_scf_plus_refinement_s', {}).get(material_key)
            if scan_bm_refined and timings_refined and all(t is not None for t in timings_refined):
                t_tot = sum(timings_refined) / 1000.0; v0_refi, e0_refi, b0_refi, b1_refi = (scan_bm_refined['min_volume']/scan_scaling_factor, scan_bm_refined['E0']/scan_scaling_factor, scan_bm_refined['bulk_modulus_ev_ang3'], scan_bm_refined['bulk_deriv'])
                summary_data['delta']['refined'].append((t_tot, delta(v0_refi, b0_refi, b1_refi, v0_ref_fu, b0_ref, b1_ref, 1.0, 0, 0)))
                summary_data['nu']['refined'].append((t_tot, nu(v0_refi, b0_refi, b1_refi, v0_ref_fu, b0_ref, b1_ref, PREFACTOR, NU_PRIME_WEIGHT_B0, NU_PRIME_WEIGHT_B1)))
                summary_data['epsilon']['refined'].append((t_tot, epsilon(v0_refi, b0_refi, b1_refi, v0_ref_fu, b0_ref, b1_ref, PREFACTOR, 0, 0)))

    print("\nGenerating summary plots...")
    all_times = np.concatenate([np.array(d)[:, 0] for m in summary_data for d in summary_data[m].values() if d])
    if len(all_times) < 2: sys.exit("Not enough data to generate summary plots.")
    min_time, max_time = np.min(all_times), np.max(all_times)
    time_bins = np.linspace(min_time, max_time, 6)
    
    binned_data = {m: {'preref': [[] for _ in time_bins[:-1]], 'refined': [[] for _ in time_bins[:-1]]} for m in summary_data}
    for metric in summary_data:
        for time, error in summary_data[metric]['preref']:
            for i in range(len(time_bins) - 1):
                if time_bins[i] <= time < time_bins[i+1]: binned_data[metric]['preref'][i].append(error); break
        for time, error in summary_data[metric]['refined']:
            for i in range(len(time_bins) - 1):
                if time_bins[i] <= time < time_bins[i+1]: binned_data[metric]['refined'][i].append(error); break
    
    positions = np.arange(len(time_bins) - 1)
    labels = [f"{time_bins[i]:.1f}-{time_bins[i+1]:.1f}" for i in range(len(time_bins) - 1)]
    
    plot_params = {
        'delta': (r'$\Delta$ [meV/atom]', r'$\Delta$', THRESHOLDS['delta']),
        'nu': (r'$\nu$ [ ]', r'$\nu$', THRESHOLDS['nu']),
        'epsilon': (r'$\varepsilon$ [ ]', r'$\varepsilon$', THRESHOLDS['epsilon'])
    }
    
    for metric, (ylabel, title_suffix, thresholds) in plot_params.items():
        fig, ax = pl.subplots(figsize=(4.0, 3.5))
        
        split_violin_plot(ax, binned_data[metric]['preref'], binned_data[metric]['refined'], positions)
        
        ax.set_yscale('log'); ax.set_ylabel(ylabel)
        ax.set_xlabel(r'Wall-Clock Time Bins [$10^3$ s]')
        ax.set_title(f'Distribution of {title_suffix} Error', pad=8)
        
        if thresholds:
            ax.axhline(thresholds['good'], linestyle='dotted', color='black', lw=0.8)
            ax.axhline(thresholds['excellent'], linestyle='--', color='black', lw=0.8)

        ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle=':')
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='gray', alpha=0.6, label='Pre-refinement'),
                           Patch(facecolor='black', alpha=0.7, label='Refined')]
        ax.legend(handles=legend_elements, loc='best')
        fig.tight_layout()
        plot_filename = os.path.join(output_dir_summary, f"SUMMARY_violin_vs_time_{metric}.pdf")
        pl.savefig(plot_filename, dpi=300)
        pl.close(fig)
        print(f"Summary violin plot for {metric} saved to '{plot_filename}'.")

    print("\nAll summary violin plots generated.")
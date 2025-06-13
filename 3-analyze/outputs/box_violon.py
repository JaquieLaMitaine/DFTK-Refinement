#!/usr/bin/env python
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pylab as pl
import tqdm

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
def split_violin_plot(ax, data1, data2, positions, width=0.8):
    all_violins = []
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
    except IndexError:
        sys.exit(f"Usage: {sys.argv[0]} SET_NAME (e.g., oxides-verification-PBE-v1)")

    PLUGIN_NAME = 'dftk'

    input_dir = f"{SET_NAME}-{PLUGIN_NAME}"
    output_dir_box = os.path.join(input_dir, "timing_vs_accuracy_plots_box") 
    output_dir_violin = os.path.join(input_dir, "timing_vs_accuracy_plots_violin") 
    os.makedirs(output_dir_box, exist_ok=True)
    os.makedirs(output_dir_violin, exist_ok=True)
    
    ref_results_file = os.path.join(input_dir, "results_2high_ecut.json")

    try:
        with open(ref_results_file, 'r') as f: ref_data = json.load(f)
    except FileNotFoundError: sys.exit(f"ERROR: High-Ecut reference data file not found at: {ref_results_file}")

    ECUT_FACTORS = [0.025, 0.05, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275, 0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.5]
    all_scan_data = {}
    for factor in ECUT_FACTORS:
        try:
            with open(os.path.join(input_dir, f"results_{factor}.json"), 'r') as f: all_scan_data[factor] = json.load(f)
        except FileNotFoundError: print(f"Warning: Scan data file for factor {factor} not found, skipping.")

    summary_data = { p: {'preref': [], 'refined': []} for p in ['v0', 'e0', 'b0', 'b1'] }
    all_materials_in_ref = sorted(list(ref_data.get('BM_fit_data', {}).keys()))

    progress_bar = tqdm.tqdm(all_materials_in_ref, desc="Processing materials")
    for material_key in progress_bar:
        progress_bar.set_description(f"{material_key:12s}")
        ref_bm_fit = ref_data.get('BM_fit_data', {}).get(material_key)
        ref_natoms = ref_data.get('num_atoms_in_sim_cell', {}).get(material_key)
        if not ref_bm_fit or ref_natoms is None: continue
        v0_ref_per_atom = ref_bm_fit['min_volume'] / ref_natoms
        e0_ref_per_atom = ref_bm_fit['E0'] / ref_natoms
        b0_ref = ref_bm_fit['bulk_modulus_ev_ang3']
        b1_ref = ref_bm_fit['bulk_deriv']
        for factor in ECUT_FACTORS:
            scan_data = all_scan_data.get(factor)
            if not scan_data: continue
            scan_bm_fit_preref = scan_data.get('BM_fit_data_pre_refinement', {}).get(material_key)
            scan_bm_fit_refined = scan_data.get('BM_fit_data', {}).get(material_key)
            scan_natoms = scan_data.get('num_atoms_in_sim_cell', {}).get(material_key)
            timings_preref = scan_data.get('timings_scf_s', {}).get(material_key)
            timings_refined = scan_data.get('timings_scf_plus_refinement_s', {}).get(material_key)
            if scan_natoms is None: continue
            if scan_bm_fit_preref and timings_preref and all(t is not None for t in timings_preref):
                t_tot = sum(timings_preref) / 1000.0
                err_v0 = abs((scan_bm_fit_preref['min_volume'] / scan_natoms) - v0_ref_per_atom); summary_data['v0']['preref'].append((t_tot, err_v0))
                err_e0 = abs((scan_bm_fit_preref['E0'] / scan_natoms) - e0_ref_per_atom); summary_data['e0']['preref'].append((t_tot, err_e0))
                err_b0 = abs(scan_bm_fit_preref['bulk_modulus_ev_ang3'] - b0_ref); summary_data['b0']['preref'].append((t_tot, err_b0))
                err_b1 = abs(scan_bm_fit_preref['bulk_deriv'] - b1_ref); summary_data['b1']['preref'].append((t_tot, err_b1))
            if scan_bm_fit_refined and timings_refined and all(t is not None for t in timings_refined):
                t_tot = sum(timings_refined) / 1000.0
                err_v0 = abs((scan_bm_fit_refined['min_volume'] / scan_natoms) - v0_ref_per_atom); summary_data['v0']['refined'].append((t_tot, err_v0))
                err_e0 = abs((scan_bm_fit_refined['E0'] / scan_natoms) - e0_ref_per_atom); summary_data['e0']['refined'].append((t_tot, err_e0))
                err_b0 = abs(scan_bm_fit_refined['bulk_modulus_ev_ang3'] - b0_ref); summary_data['b0']['refined'].append((t_tot, err_b0))
                err_b1 = abs(scan_bm_fit_refined['bulk_deriv'] - b1_ref); summary_data['b1']['refined'].append((t_tot, err_b1))

    print("\nGenerating summary plots...")

    all_times = np.concatenate([np.array(d)[:, 0] for p in summary_data for d in summary_data[p].values() if d])
    if len(all_times) < 2:
        print("Not enough data to generate summary plots.")
        sys.exit()
    min_time, max_time = np.min(all_times), np.max(all_times)
    time_bins = np.linspace(min_time, max_time, 6)
    
    binned_data = {p: {'preref': [[] for _ in time_bins[:-1]], 'refined': [[] for _ in time_bins[:-1]]} for p in summary_data}
    for param in summary_data:
        for time, error in summary_data[param]['preref']:
            for i in range(len(time_bins) - 1):
                if time_bins[i] <= time < time_bins[i+1]: binned_data[param]['preref'][i].append(error); break
        for time, error in summary_data[param]['refined']:
            for i in range(len(time_bins) - 1):
                if time_bins[i] <= time < time_bins[i+1]: binned_data[param]['refined'][i].append(error); break
    
    positions = np.arange(len(time_bins) - 1)
    labels = [f"{time_bins[i]:.1f}-{time_bins[i+1]:.1f}" for i in range(len(time_bins) - 1)]
    
    def create_summary_boxplot_superposed(param_name, y_label, title_suffix):
        fig, ax = pl.subplots(figsize=(3.3, 3.0))
        boxprops_preref = {'facecolor': (0.7, 0.7, 0.7, 0.6), 'edgecolor': 'gray', 'linestyle': '--'}
        medianprops_preref = {'color': 'gray', 'linestyle': '--', 'lw': 1.5}
        boxprops_refined = {'facecolor': (0, 0, 0, 0.5), 'edgecolor': 'black'}
        medianprops_refined = {'color': 'lightgray', 'lw': 1.5}
        
        ax.boxplot(binned_data[param_name]['preref'], positions=positions, widths=0.6, patch_artist=True,
                   boxprops=boxprops_preref, medianprops=medianprops_preref,
                   whiskerprops=dict(color='gray', linestyle='--'), capprops=dict(color='gray', linestyle='--'), showfliers=False)
        ax.boxplot(binned_data[param_name]['refined'], positions=positions, widths=0.35, patch_artist=True,
                   boxprops=boxprops_refined, medianprops=medianprops_refined,
                   whiskerprops=dict(color='black'), capprops=dict(color='black'), showfliers=False)
        
        ax.set_yscale('log'); ax.set_ylabel(y_label)
        ax.set_xlabel(r'Wall-Clock Time Bins [$10^3$ s]')
        ax.set_title(f'Distribution of {title_suffix} Error', pad=8)
        ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle=':')
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=(0.7, 0.7, 0.7, 0.6), edgecolor='gray', linestyle='--', label='Pre-refinement'),
                           Patch(facecolor=(0, 0, 0, 0.5), edgecolor='black', label='Refined')]
        ax.legend(handles=legend_elements, loc='best')
        fig.tight_layout()
        plot_filename = os.path.join(output_dir_box, f"SUMMARY_boxplot_superposed_{param_name}_error.pdf")
        pl.savefig(plot_filename, dpi=300); pl.close(fig)
        print(f"Summary boxplot for {param_name} saved to '{plot_filename}'.")

    def create_summary_violin_plot(param_name, y_label, title_suffix):
        fig, ax = pl.subplots(figsize=(3.3, 3.0))
        split_violin_plot(ax, binned_data[param_name]['preref'], binned_data[param_name]['refined'], positions)
        
        ax.set_yscale('log'); ax.set_ylabel(y_label)
        ax.set_xlabel(r'Wall-Clock Time Bins [$10^3$ s]')
        ax.set_title(f'Distribution of {title_suffix} Error', pad=8)
        ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle=':')
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='gray', alpha=0.6, label='Pre-refinement'),
                           Patch(facecolor='black', alpha=0.7, label='Refined')]
        ax.legend(handles=legend_elements, loc='best')
        fig.tight_layout()
        plot_filename = os.path.join(output_dir_violin, f"SUMMARY_violin_{param_name}_error.pdf")
        pl.savefig(plot_filename, dpi=300); pl.close(fig)
        print(f"Summary violin plot for {param_name} saved to '{plot_filename}'.")

    plot_params = {
        'v0': (r'$|V_0 - V_{0, \mathrm{ref}}|$ [\AA³/atom]', '$V_0$'),
        'e0': (r'$|E_0 - E_{0, \mathrm{ref}}|$ [eV/atom]', '$E_0$'),
        'b0': (r'$|B_0 - B_{0, \mathrm{ref}}|$ [eV/\AA³]', '$B_0$'),
        'b1': (r"$|B'_0 - B'_{0, \mathrm{ref}}|$ [ ]", "$B_0'$")
    }
    for param, (ylabel, title_suffix) in plot_params.items():
        create_summary_boxplot_superposed(param, ylabel, title_suffix)
        create_summary_violin_plot(param, ylabel, title_suffix)
    
    print("\nAll summary plots generated.")
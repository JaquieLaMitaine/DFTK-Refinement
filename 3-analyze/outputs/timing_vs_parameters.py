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
# ---

if __name__ == "__main__":
    try:
        SET_NAME = sys.argv[1]
    except IndexError:
        sys.exit(f"Usage: {sys.argv[0]} SET_NAME (e.g., oxides-verification-PBE-v1)")

    PLUGIN_NAME = 'dftk'

    input_dir = f"{SET_NAME}-{PLUGIN_NAME}"
    output_dir = os.path.join(input_dir, "timing_vs_accuracy_plots_linear_time_ks") 
    os.makedirs(output_dir, exist_ok=True)
    
    ref_results_file = os.path.join(input_dir, "results_2high_ecut.json")

    # --- Load Data ---
    try:
        with open(ref_results_file, 'r') as f:
            ref_data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"ERROR: High-Ecut reference data file not found at: {ref_results_file}")

    ECUT_FACTORS = [0.025, 0.05, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275, 0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.5]

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
        'v0': {'preref': [], 'refined': []},
        'e0': {'preref': [], 'refined': []},
        'b0': {'preref': [], 'refined': []},
        'b1': {'preref': [], 'refined': []},
    }

    # --- Main processing loop ---
    all_materials_in_ref = sorted(list(ref_data.get('BM_fit_data', {}).keys()))

    progress_bar = tqdm.tqdm(all_materials_in_ref, desc="Processing materials")
    for material_key in progress_bar:
        progress_bar.set_description(f"{material_key:12s}")

        ref_bm_fit = ref_data.get('BM_fit_data', {}).get(material_key)
        ref_natoms = ref_data.get('num_atoms_in_sim_cell', {}).get(material_key)
        if not ref_bm_fit or ref_natoms is None:
            continue
        
        v0_ref_per_atom = ref_bm_fit['min_volume'] / ref_natoms
        e0_ref_per_atom = ref_bm_fit['E0'] / ref_natoms
        b0_ref = ref_bm_fit['bulk_modulus_ev_ang3']
        b1_ref = ref_bm_fit['bulk_deriv']

        conv_v0_preref, conv_v0_refined = [], []
        conv_e0_preref, conv_e0_refined = [], []
        conv_b0_preref, conv_b0_refined = [], []
        conv_b1_preref, conv_b1_refined = [], []

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
                err_v0 = abs((scan_bm_fit_preref['min_volume'] / scan_natoms) - v0_ref_per_atom)
                err_e0 = abs((scan_bm_fit_preref['E0'] / scan_natoms) - e0_ref_per_atom)
                err_b0 = abs(scan_bm_fit_preref['bulk_modulus_ev_ang3'] - b0_ref)
                err_b1 = abs(scan_bm_fit_preref['bulk_deriv'] - b1_ref)
                conv_v0_preref.append((t_tot, err_v0)); summary_data['v0']['preref'].append((t_tot, err_v0))
                conv_e0_preref.append((t_tot, err_e0)); summary_data['e0']['preref'].append((t_tot, err_e0))
                conv_b0_preref.append((t_tot, err_b0)); summary_data['b0']['preref'].append((t_tot, err_b0))
                conv_b1_preref.append((t_tot, err_b1)); summary_data['b1']['preref'].append((t_tot, err_b1))

            if scan_bm_fit_refined and timings_refined and all(t is not None for t in timings_refined):
                t_tot = sum(timings_refined) / 1000.0
                err_v0 = abs((scan_bm_fit_refined['min_volume'] / scan_natoms) - v0_ref_per_atom)
                err_e0 = abs((scan_bm_fit_refined['E0'] / scan_natoms) - e0_ref_per_atom)
                err_b0 = abs(scan_bm_fit_refined['bulk_modulus_ev_ang3'] - b0_ref)
                err_b1 = abs(scan_bm_fit_refined['bulk_deriv'] - b1_ref)
                conv_v0_refined.append((t_tot, err_v0)); summary_data['v0']['refined'].append((t_tot, err_v0))
                conv_e0_refined.append((t_tot, err_e0)); summary_data['e0']['refined'].append((t_tot, err_e0))
                conv_b0_refined.append((t_tot, err_b0)); summary_data['b0']['refined'].append((t_tot, err_b0))
                conv_b1_refined.append((t_tot, err_b1)); summary_data['b1']['refined'].append((t_tot, err_b1))

        def make_plot(ax, data_preref, data_refined, ylabel, title):
            data_preref.sort(); data_refined.sort()
            if data_preref:
                times_preref, errors_preref = zip(*data_preref)
                ax.plot(times_preref, errors_preref, 's--', mfc='none', color='black', label='Pre-refinement', markersize=3, linewidth=0.8)
            if data_refined:
                times_refined, errors_refined = zip(*data_refined)
                ax.plot(times_refined, errors_refined, 'o-', mfc='none', color='black', label='Refined', markersize=3, linewidth=1.0)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel(ylabel); ax.grid(True, linestyle=':'); ax.legend(loc='upper right'); ax.set_title(title, pad=8)

        all_plots_data = {
            'V0': (conv_v0_preref, conv_v0_refined, r"$|V_0 - V_{0, \mathrm{ref}}|$ [\AA³/atom]"),
            'E0': (conv_e0_preref, conv_e0_refined, r"$|E_0 - E_{0, \mathrm{ref}}|$ [eV/atom]"),
            'B0': (conv_b0_preref, conv_b0_refined, r"$|B_0 - B_{0, \mathrm{ref}}|$ [eV/\AA³]"),
            'B1': (conv_b1_preref, conv_b1_refined, r"$|B'_0 - B'_{0, \mathrm{ref}}|$ [ ]")
        }
        for param, (data_pre, data_ref, ylabel) in all_plots_data.items():
            if data_pre or data_ref:
                fig, ax = pl.subplots(figsize=(3.3, 3.0))
                title = f"${param}$ Convergence for {material_key.replace('_', '-')}"
                make_plot(ax, data_pre, data_ref, ylabel, title)
                ax.set_xlabel(r'Total Wall-Clock Time [$10^3$ s]')
                fig.tight_layout()
                plot_filename = os.path.join(output_dir, f"{material_key}_{param}_conv_vs_time.pdf")
                pl.savefig(plot_filename, dpi=300)
                pl.close(fig)

    print(f"\nIndividual plots generated in '{output_dir}'.")
    print("\nGenerating summary box-plots (grouped by wall-clock time)...")

    def create_summary_boxplot_vs_time(param_name, y_label, title_suffix):
        all_data_preref = np.array(summary_data[param_name]['preref'])
        all_data_refined = np.array(summary_data[param_name]['refined'])
        if len(all_data_preref) == 0 and len(all_data_refined) == 0: return

        all_times = np.concatenate([all_data_preref[:, 0], all_data_refined[:, 0]])
        if len(all_times) < 2: return
        
        min_time, max_time = np.min(all_times), np.max(all_times)
        min_log_time = np.log10(np.min(all_times[all_times > 0])) # Eviter log(0)
        max_log_time = np.log10(np.max(all_times))
        time_bins = np.logspace(min_log_time, max_log_time, 10)
        
        binned_data_preref = [[] for _ in range(len(time_bins) - 1)]
        binned_data_refined = [[] for _ in range(len(time_bins) - 1)]

        for time, error in all_data_preref:
            for i in range(len(time_bins) - 1):
                if time_bins[i] <= time < time_bins[i+1]: binned_data_preref[i].append(error); break
        
        for time, error in all_data_refined:
            for i in range(len(time_bins) - 1):
                if time_bins[i] <= time < time_bins[i+1]: binned_data_refined[i].append(error); break

        positions = np.arange(len(time_bins) - 1)
        labels = [f"{time_bins[i]:.1f}-{time_bins[i+1]:.1f}" for i in range(len(time_bins) - 1)]

        fig, ax = pl.subplots(figsize=(4.0, 3.5))
        boxprops_preref = {'facecolor': 'none', 'edgecolor': 'gray'}
        medianprops_preref = {'color': 'gray'}
        boxprops_refined = {'facecolor': 'none', 'edgecolor': 'black'}
        medianprops_refined = {'color': 'black'}
        ax.boxplot(binned_data_preref, positions=positions, widths=0.35, patch_artist=True,
                   boxprops=boxprops_preref, medianprops=medianprops_preref,
                   whiskerprops=dict(color='gray'), capprops=dict(color='gray'), showfliers=False)
        ax.boxplot(binned_data_refined, positions=positions, widths=0.35, patch_artist=True,
                   boxprops=boxprops_refined, medianprops=medianprops_refined,
                   whiskerprops=dict(color='black'), capprops=dict(color='black'), showfliers=False)
        
        ax.set_yscale('log')
        ax.set_ylabel(y_label)
        ax.set_xlabel(r'Wall-Clock Time Bins [$10^3$ s]')
        ax.set_title(f'Distribution of {title_suffix} Error', pad=8)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle=':')
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='gray', lw=2, label='Pre-refinement'),
                           Line2D([0], [0], color='black', lw=2, label='Refined')]
        ax.legend(handles=legend_elements, loc='best')
        fig.tight_layout()
        plot_filename = os.path.join(output_dir, f"SUMMARY_boxplot_vs_time_linear_{param_name}_error.pdf")
        pl.savefig(plot_filename, dpi=300)
        pl.close(fig)
        print(f"Summary boxplot for {param_name} saved to '{plot_filename}'.")

    create_summary_boxplot_vs_time('v0', r'$|V_0 - V_{0, \mathrm{ref}}|$ [\AA³/atom]', '$V_0$')
    create_summary_boxplot_vs_time('e0', r'$|E_0 - E_{0, \mathrm{ref}}|$ [eV/atom]', '$E_0$')
    create_summary_boxplot_vs_time('b0', r'$|B_0 - B_{0, \mathrm{ref}}|$ [eV/\AA³]', '$B_0$')
    create_summary_boxplot_vs_time('b1', r"$|B'_0 - B'_{0, \mathrm{ref}}|$ [ ]", "$B_0'$")
    
    print("\nAll summary boxplots generated.")
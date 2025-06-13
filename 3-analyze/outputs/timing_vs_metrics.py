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

    # --- Config ---
    NU_PRIME_WEIGHT_B0 = 5.7381
    NU_PRIME_WEIGHT_B1 = 1.5522
    PREFACTOR = 1.0
    THRESHOLDS = {
        'delta': {'excellent': 0.3, 'good': 0.95},
        'nu': {'excellent': 0.10, 'good': 0.33},
        'epsilon': {'excellent': 0.06, 'good': 0.20},
    }
    ECUT_FACTORS = [0.025, 0.05, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275, 0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.5]

    # --- Define file paths ---
    input_dir = f"{SET_NAME}-{PLUGIN_NAME}"
    output_dir = os.path.join(input_dir, "timing_vs_metrics_plots")
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
        'delta': {'preref': [], 'refined': []},
        'nu': {'preref': [], 'refined': []},
        'epsilon': {'preref': [], 'refined': []},
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
            
        element, configuration = material_key.split('-')
        ref_scaling_factor = get_volume_scaling_to_formula_unit(ref_natoms, element, configuration)
        v0_ref_fu = ref_bm_fit['min_volume'] / ref_scaling_factor
        e0_ref_fu = ref_bm_fit['E0'] / ref_scaling_factor
        b0_ref = ref_bm_fit['bulk_modulus_ev_ang3']
        b1_ref = ref_bm_fit['bulk_deriv']

        conv_delta_preref, conv_delta_refined = [], []
        conv_nu_preref, conv_nu_refined = [], []
        conv_epsilon_preref, conv_epsilon_refined = [], []

        for factor in ECUT_FACTORS:
            scan_data = all_scan_data.get(factor)
            if not scan_data: continue

            scan_natoms = scan_data.get('num_atoms_in_sim_cell', {}).get(material_key)
            if scan_natoms is None: continue
            scan_scaling_factor = get_volume_scaling_to_formula_unit(scan_natoms, element, configuration)

            scan_bm_fit_preref = scan_data.get('BM_fit_data_pre_refinement', {}).get(material_key)
            timings_preref = scan_data.get('timings_scf_s', {}).get(material_key)
            if scan_bm_fit_preref and timings_preref and all(t is not None for t in timings_preref):
                t_tot = sum(timings_preref) / 1000.0
                v0_scan_pre_fu = scan_bm_fit_preref['min_volume'] / scan_scaling_factor
                e0_scan_pre_fu = scan_bm_fit_preref['E0'] / scan_scaling_factor
                b0_scan_pre = scan_bm_fit_preref['bulk_modulus_ev_ang3']
                b1_scan_pre = scan_bm_fit_preref['bulk_deriv']
                
                delta_val = delta(v0_scan_pre_fu, b0_scan_pre, b1_scan_pre, v0_ref_fu, b0_ref, b1_ref, 1.0, 0, 0)
                nu_val = nu(v0_scan_pre_fu, b0_scan_pre, b1_scan_pre, v0_ref_fu, b0_ref, b1_ref, PREFACTOR, NU_PRIME_WEIGHT_B0, NU_PRIME_WEIGHT_B1)
                eps_val = epsilon(v0_scan_pre_fu, b0_scan_pre, b1_scan_pre, v0_ref_fu, b0_ref, b1_ref, PREFACTOR, 0, 0)

                conv_delta_preref.append((t_tot, delta_val)); summary_data['delta']['preref'].append((t_tot, delta_val))
                conv_nu_preref.append((t_tot, nu_val)); summary_data['nu']['preref'].append((t_tot, nu_val))
                conv_epsilon_preref.append((t_tot, eps_val)); summary_data['epsilon']['preref'].append((t_tot, eps_val))

            scan_bm_fit_refined = scan_data.get('BM_fit_data', {}).get(material_key)
            timings_refined = scan_data.get('timings_scf_plus_refinement_s', {}).get(material_key)
            if scan_bm_fit_refined and timings_refined and all(t is not None for t in timings_refined):
                t_tot = sum(timings_refined) / 1000.0
                v0_scan_ref_fu = scan_bm_fit_refined['min_volume'] / scan_scaling_factor
                e0_scan_ref_fu = scan_bm_fit_refined['E0'] / scan_scaling_factor
                b0_scan_ref = scan_bm_fit_refined['bulk_modulus_ev_ang3']
                b1_scan_ref = scan_bm_fit_refined['bulk_deriv']

                delta_val = delta(v0_scan_ref_fu, b0_scan_ref, b1_scan_ref, v0_ref_fu, b0_ref, b1_ref, 1.0, 0, 0)
                nu_val = nu(v0_scan_ref_fu, b0_scan_ref, b1_scan_ref, v0_ref_fu, b0_ref, b1_ref, PREFACTOR, NU_PRIME_WEIGHT_B0, NU_PRIME_WEIGHT_B1)
                eps_val = epsilon(v0_scan_ref_fu, b0_scan_ref, b1_scan_ref, v0_ref_fu, b0_ref, b1_ref, PREFACTOR, 0, 0)

                conv_delta_refined.append((t_tot, delta_val)); summary_data['delta']['refined'].append((t_tot, delta_val))
                conv_nu_refined.append((t_tot, nu_val)); summary_data['nu']['refined'].append((t_tot, nu_val))
                conv_epsilon_refined.append((t_tot, eps_val)); summary_data['epsilon']['refined'].append((t_tot, eps_val))
        
        def make_plot(ax, data_preref, data_refined, ylabel, title, thresholds):
            data_preref.sort(); data_refined.sort()
            if data_preref:
                times_preref, errors_preref = zip(*data_preref)
                ax.plot(times_preref, errors_preref, 's--', mfc='none', color='black', label='Pre-refinement', markersize=3, linewidth=0.8)
            if data_refined:
                times_refined, errors_refined = zip(*data_refined)
                ax.plot(times_refined, errors_refined, 'o-', mfc='none', color='black', label='Refined', markersize=3, linewidth=1.0)
            
            ax.set_yscale('log'); ax.set_ylabel(ylabel); ax.grid(True, linestyle=':')
            ax.legend(loc='upper right'); ax.set_title(title, pad=8)
            if thresholds:
                ax.axhline(thresholds['good'], linestyle='dotted', color='black', lw=0.8)
                ax.axhline(thresholds['excellent'], linestyle='--', color='black', lw=0.8)

        all_plots_data = {
            'delta': (conv_delta_preref, conv_delta_refined, r'$\Delta$ [meV/atom]', THRESHOLDS['delta']),
            'nu': (conv_nu_preref, conv_nu_refined, r'$\nu$ [ ]', THRESHOLDS['nu']),
            'epsilon': (conv_epsilon_preref, conv_epsilon_refined, r'$\varepsilon$ [ ]', THRESHOLDS['epsilon'])
        }
        for metric, (data_pre, data_ref, ylabel, thresholds) in all_plots_data.items():
            if data_pre or data_ref:
                fig, ax = pl.subplots(figsize=(3.3, 3.0))
                title = f"$\\{metric}$ Convergence for {material_key.replace('_', '-')}"
                make_plot(ax, data_pre, data_ref, ylabel, title, thresholds)
                ax.set_xlabel(r'Total Wall-Clock Time [$10^3$ s]')
                fig.tight_layout()
                plot_filename = os.path.join(output_dir, f"{material_key}_{metric}_conv_vs_time.pdf")
                pl.savefig(plot_filename, dpi=300)
                pl.close(fig)

    print(f"\nIndividual metric plots generated in '{output_dir}'.")
    print("\nGenerating summary box-plots (grouped by wall-clock time)...")

    def create_summary_boxplot_vs_time(metric_name, y_label, title_suffix, thresholds):
        all_data_preref = np.array(summary_data[metric_name]['preref'])
        all_data_refined = np.array(summary_data[metric_name]['refined'])
        if len(all_data_preref) == 0 and len(all_data_refined) == 0: return

        all_times = np.concatenate([all_data_preref[:, 0], all_data_refined[:, 0]])
        if len(all_times) < 2: return
        
        min_time, max_time = np.min(all_times), np.max(all_times)
        min_log_time = np.log10(np.min(all_times[all_times > 0])) # Eviter log(0)
        max_log_time = np.log10(np.max(all_times))
        time_bins = np.logspace(min_log_time, max_log_time, 10)
        
        binned_data_preref = [[] for _ in range(len(time_bins) - 1)]
        binned_data_refined = [[] for _ in range(len(time_bins) - 1)]
        for t, e in all_data_preref:
            for i in range(len(time_bins) - 1):
                if time_bins[i] <= t < time_bins[i+1]: binned_data_preref[i].append(e); break
        for t, e in all_data_refined:
            for i in range(len(time_bins) - 1):
                if time_bins[i] <= t < time_bins[i+1]: binned_data_refined[i].append(e); break

        positions = np.arange(len(time_bins) - 1)
        labels = [f"{time_bins[i]:.1f}-{time_bins[i+1]:.1f}" for i in range(len(time_bins) - 1)]

        fig, ax = pl.subplots(figsize=(3.3, 3.0))
        boxprops_preref = {'facecolor': 'none', 'edgecolor': 'gray'}
        medianprops_preref = {'color': 'gray',}
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
        ax.set_title(f'Distribution of {title_suffix}', pad=8)
        if thresholds:
            ax.axhline(thresholds['good'], linestyle='dotted', color='black', lw=0.8)
            ax.axhline(thresholds['excellent'], linestyle='--', color='black', lw=0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle=':')
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='gray', linestyle='--', lw=2, label='Pre-refinement'),
                           Line2D([0], [0], color='black', lw=2, label='Refined')]
        if thresholds:
            legend_elements.append(Line2D([0], [0], color='black', linestyle='dotted', lw=0.8, label='Good'))
            legend_elements.append(Line2D([0], [0], color='black', linestyle='--', lw=0.8, label='Excellent'))
        ax.legend(handles=legend_elements, loc='best')
        fig.tight_layout()
        plot_filename = os.path.join(output_dir, f"SUMMARY_boxplot_vs_time_{metric_name}.pdf")
        pl.savefig(plot_filename, dpi=300)
        pl.close(fig)
        print(f"Summary boxplot for {metric_name} saved to '{plot_filename}'.")

    create_summary_boxplot_vs_time('delta', r'$\Delta$ [meV/atom]', r'$\Delta$', THRESHOLDS['delta'])
    create_summary_boxplot_vs_time('nu', r'$\nu$ [ ]', r'$\nu$', THRESHOLDS['nu'])
    create_summary_boxplot_vs_time('epsilon', r'$\varepsilon$ [ ]', r'$\varepsilon$', THRESHOLDS['epsilon'])
    
    print("\nAll summary boxplots generated.")
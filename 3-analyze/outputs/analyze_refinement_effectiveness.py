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

# --- Configuration ---
# <<< CHANGEMENT: Facteurs sur lesquels on va moyenner le score >>>
ECUT_FACTORS_TO_AVERAGE = [0.225, 0.250, 0.275]
NU_PRIME_WEIGHT_B0 = 5.7381
NU_PRIME_WEIGHT_B1 = 1.5522
PREFACTOR = 1.0
# ---

if __name__ == "__main__":
    try:
        SET_NAME = sys.argv[1]
    except IndexError:
        sys.exit(f"Usage: {sys.argv[0]} SET_NAME (e.g., oxides-verification-PBE-v1)")

    PLUGIN_NAME = 'dftk'

    input_dir = f"{SET_NAME}-{PLUGIN_NAME}"
    output_dir = os.path.join(input_dir, f"effectiveness_analysis_averaged")
    os.makedirs(output_dir, exist_ok=True)
    
    ref_results_file = os.path.join(input_dir, "results_2high_ecut.json")

    try:
        with open(ref_results_file, 'r') as f: ref_data = json.load(f)
    except FileNotFoundError: sys.exit(f"ERROR: High-Ecut reference data file not found at: {ref_results_file}")
    
    all_scan_data = {}
    for factor in ECUT_FACTORS_TO_AVERAGE:
        scan_results_file = os.path.join(input_dir, f"results_{factor}.json")
        try:
            with open(scan_results_file, 'r') as f:
                all_scan_data[factor] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Scan data for factor {factor} not found. It will be ignored in the average.")

    if not all_scan_data:
        sys.exit("ERROR: No scan data found for any of the specified factors. Cannot proceed.")

    analysis_data = []

    all_materials = sorted(list(ref_data.get('BM_fit_data', {}).keys()))

    progress_bar = tqdm.tqdm(all_materials, desc="Analyzing materials")
    for material_key in progress_bar:
        ref_bm_fit = ref_data.get('BM_fit_data', {}).get(material_key)
        ref_natoms = ref_data.get('num_atoms_in_sim_cell', {}).get(material_key)
        if not ref_bm_fit or ref_natoms is None: continue
        
        element, configuration = material_key.split('-')
        ref_scaling_factor = get_volume_scaling_to_formula_unit(ref_natoms, element, configuration)
        v0_ref_fu = ref_bm_fit['min_volume'] / ref_scaling_factor
        e0_ref_fu = ref_bm_fit['E0'] / ref_scaling_factor
        b0_ref = ref_bm_fit['bulk_modulus_ev_ang3']
        b1_ref = ref_bm_fit['bulk_deriv']

        scores_delta, scores_nu, scores_epsilon = [], [], []

        for factor in ECUT_FACTORS_TO_AVERAGE:
            scan_data = all_scan_data.get(factor)
            if not scan_data: continue

            scan_bm_preref = scan_data.get('BM_fit_data_pre_refinement', {}).get(material_key)
            scan_bm_refined = scan_data.get('BM_fit_data', {}).get(material_key)
            scan_natoms = scan_data.get('num_atoms_in_sim_cell', {}).get(material_key)
            if not all([scan_bm_preref, scan_bm_refined, scan_natoms]): continue

            scan_scaling_factor = get_volume_scaling_to_formula_unit(scan_natoms, element, configuration)

            v0_pre, e0_pre, b0_pre, b1_pre = (scan_bm_preref['min_volume']/scan_scaling_factor, scan_bm_preref['E0']/scan_scaling_factor, scan_bm_preref['bulk_modulus_ev_ang3'], scan_bm_preref['bulk_deriv'])
            delta_preref = delta(v0_pre, b0_pre, b1_pre, v0_ref_fu, b0_ref, b1_ref, 1.0, 0, 0)
            nu_preref = nu(v0_pre, b0_pre, b1_pre, v0_ref_fu, b0_ref, b1_ref, PREFACTOR, NU_PRIME_WEIGHT_B0, NU_PRIME_WEIGHT_B1)
            epsilon_preref = epsilon(v0_pre, b0_pre, b1_pre, v0_ref_fu, b0_ref, b1_ref, PREFACTOR, 0, 0)
            
            v0_refi, e0_refi, b0_refi, b1_refi = (scan_bm_refined['min_volume']/scan_scaling_factor, scan_bm_refined['E0']/scan_scaling_factor, scan_bm_refined['bulk_modulus_ev_ang3'], scan_bm_refined['bulk_deriv'])
            delta_refined = delta(v0_refi, b0_refi, b1_refi, v0_ref_fu, b0_ref, b1_ref, 1.0, 0, 0)
            nu_refined = nu(v0_refi, b0_refi, b1_refi, v0_ref_fu, b0_ref, b1_ref, PREFACTOR, NU_PRIME_WEIGHT_B0, NU_PRIME_WEIGHT_B1)
            epsilon_refined = epsilon(v0_refi, b0_refi, b1_refi, v0_ref_fu, b0_ref, b1_ref, PREFACTOR, 0, 0)

            if delta_refined > 1e-9: scores_delta.append(delta_preref / delta_refined)
            if nu_refined > 1e-9: scores_nu.append(nu_preref / nu_refined)
            if epsilon_refined > 1e-9: scores_epsilon.append(epsilon_preref / epsilon_refined)
        
        if scores_delta and scores_nu and scores_epsilon:
            avg_score_delta = np.exp(np.mean(np.log(scores_delta)))
            avg_score_nu = np.exp(np.mean(np.log(scores_nu)))
            if avg_score_nu > 50:
                print(f"Warning: High nu score for {material_key} ({avg_score_nu}), setting to 50.")
                avg_score_nu = 50.0
            avg_score_epsilon = np.exp(np.mean(np.log(scores_epsilon)))
            
            props_scan_data = all_scan_data[ECUT_FACTORS_TO_AVERAGE[1]]
            n_electrons = props_scan_data.get('n_electrons', {}).get(material_key, 0)
            n_kpoints = props_scan_data.get('n_kpoints', {}).get(material_key, 0)

            analysis_data.append({
                'material': material_key, 'configuration': configuration,
                'score_delta': avg_score_delta, 'score_nu': avg_score_nu, 'score_epsilon': avg_score_epsilon,
                'n_electrons': n_electrons, 'n_kpoints': n_kpoints,
                'v0_ref_per_atom': ref_bm_fit['min_volume'] / ref_natoms, 'b0_ref': b0_ref
            })

    if not analysis_data:
        sys.exit("No valid data to plot after analysis.")
        
    for score_key, metric_symbol in [('score_delta', 'delta'), ('score_nu', 'nu'), ('score_epsilon', 'epsilon')]:
        fig, ax = pl.subplots(figsize=(3.3, 3.0))
        data_by_conf = defaultdict(list)
        for item in analysis_data: data_by_conf[item['configuration']].append(item[score_key])
        configs = sorted(data_by_conf.keys())
        data_to_plot = [data_by_conf[c] for c in configs]
        
        if metric_symbol == 'delta':
            symbol = r'$\Delta$'
        elif metric_symbol == 'nu':
            symbol = r'$\nu$'
        elif metric_symbol == 'epsilon':
            symbol = r'$\varepsilon$'

        ax.boxplot(data_to_plot, medianprops={'color': 'black', 'linestyle': '--', 'lw': 1.5}, labels=configs, showfliers=False)
        ax.set_ylabel(f'S({symbol})')
        ax.set_xlabel('Stoichiometry'); #ax.set_title(f'Refinement Effectiveness by Stoichiometry')
        ax.grid(True, axis='y', linestyle=':')
        pl.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fig.tight_layout()
        plot_filename = os.path.join(output_dir, f"boxplot_vs_stoichiometry_{score_key}.pdf")
        pl.savefig(plot_filename, dpi=300); pl.close(fig)

    properties_to_plot = {
        'n_electrons': r'Number of electrons', 'n_kpoints': r'Number of k-points',
        'v0_ref_per_atom': r'$V_0$ per atom [\AA³/atom]', 'b0_ref': r'$B_0$ [eV/\AA³]'
    }
    

    for score_key, metric_symbol in [('score_delta', 'delta'), ('score_nu', 'nu'), ('score_epsilon', 'epsilon')]:
        for prop_key, xlabel in properties_to_plot.items():
            if metric_symbol == 'delta': # ITs ugly, dont look at it too much
                metric_symbol = r'$\Delta$'
            elif metric_symbol == 'nu':
                metric_symbol = r'$\nu$'
            elif metric_symbol == 'epsilon':
                metric_symbol = r'$\varepsilon$'
            if prop_key == 'n_electrons':
                xlabel = r'$N_{el}$'
            elif prop_key == 'n_kpoints':
                xlabel = r'N_{k-points}'
            elif prop_key == 'v0_ref_per_atom':
                xlabel = r'$V_0$ [\AA³/atom]'
            elif prop_key == 'b0_ref':
                xlabel = r'$B_0$ [eV/\AA³]'
            fig, ax = pl.subplots(figsize=(3.3, 3.0)) 
            
            confs = sorted(list(set(d['configuration'] for d in analysis_data)))
            markers = ['s', '^', 'o', 'D', 'v', 'p', '*', 'h']
            conf_marker_map = {conf: markers[i % len(markers)] for i, conf in enumerate(confs)}

            for d in analysis_data:
                score = np.nan_to_num(d[score_key], posinf=np.max(np.isfinite([d[score_key] for d in analysis_data]))*1.5)
                
                ax.plot(d[prop_key], score,
                        marker=conf_marker_map[d['configuration']],
                        linestyle='None',
                        ms=5,
                        markerfacecolor='none',
                        markeredgecolor='black',
                        alpha=0.9)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(f'S({metric_symbol})')
            #ax.set_title(f'Refinement Score vs. {prop_key}')
            ax.grid(True, linestyle=':')
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], linestyle='None', marker=marker,
                                    label=conf, color='black', markerfacecolor='none', 
                                    markersize=5) for conf, marker in conf_marker_map.items()]

            ax.legend(handles=legend_elements, title='Config.')
            fig.tight_layout()
            plot_filename = os.path.join(output_dir, f"scatter_{score_key}_vs_{prop_key}.pdf")
            pl.savefig(plot_filename, dpi=300)
            pl.close(fig)


    print(f"\nEffectiveness analysis plots generated in '{output_dir}'.")
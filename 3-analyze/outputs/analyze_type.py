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
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})
# ---


ELECTRONIC_CHARACTER = {
   
    "BaO": "Gap > 0", "BeO": "Gap > 0", "CaO": "Gap > 0", "K2O": "Gap > 0", 
    "MgO": "Gap > 0", "PbO": "Gap > 0", "SiO": "Gap > 0", "SnO": "Gap > 0",
    "SrO": "Gap > 0", "ZnO": "Gap > 0", "GeO": "Gap > 0", "Be2O": "Gap > 0",
    "Ca2O": "Gap > 0", "Cs2O": "Gap > 0", "H2O": "Gap > 0", "Li2O": "Gap > 0",
    "Mg2O": "Gap > 0", "Na2O": "Gap > 0", "Rb2O": "Gap > 0", "Sr2O": "Gap > 0",
    "Tl2O": "Gap > 0", "Zn2O": "Gap > 0", "HfO2": "Gap > 0", "ZrO2": "Gap > 0",
    "SeO2": "Gap > 0", "SiO2": "Gap > 0", "TeO2": "Gap > 0", "TiO2": "Gap > 0",
    "WO3": "Gap > 0", "PoO2": "Gap > 0", "Ir2O3": "Gap > 0", "FeO2": "Gap > 0",
    
    # M: metallic at 0 K
    "RuO2": "Metallic", "OsO2": "Metallic",
    
    # U: unclassified / no reliable low-T data
    "RnO3": "Unclassified",
}

def get_chemical_formula(element, configuration):
    """
    Crée la formule chimique standard à partir de l'élément et de la configuration.
    """
    if 'O' not in configuration:
        return element
    
    if configuration == 'XO' and element in ['K', 'Rb', 'Cs', 'Na', 'Li', 'H', 'Tl']:
        return f"{element}2O"
        
    return configuration.replace('X', element)

# --- Configuration ---
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
    output_dir = os.path.join(input_dir, f"effectiveness_analysis_by_type")
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
        
        formula = get_chemical_formula(element, configuration)
        electronic_category = ELECTRONIC_CHARACTER.get(formula.replace('*',''), 'Unclassified') # Default to Unclassified
        
        ref_scaling_factor = get_volume_scaling_to_formula_unit(ref_natoms, element, configuration)
        v0_ref_fu, e0_ref_fu, b0_ref, b1_ref = (
            ref_bm_fit['min_volume'] / ref_scaling_factor, ref_bm_fit['E0'] / ref_scaling_factor,
            ref_bm_fit['bulk_modulus_ev_ang3'], ref_bm_fit['bulk_deriv']
        )
        
        scores_delta, scores_nu, scores_epsilon = [], [], []
        for factor in ECUT_FACTORS_TO_AVERAGE:
            scan_data = all_scan_data.get(factor)
            if not scan_data: continue

            scan_bm_preref = scan_data.get('BM_fit_data_pre_refinement', {}).get(material_key)
            scan_bm_refined = scan_data.get('BM_fit_data', {}).get(material_key)
            scan_natoms = scan_data.get('num_atoms_in_sim_cell', {}).get(material_key)
            if not all([scan_bm_preref, scan_bm_refined, scan_natoms]): continue

            scan_scaling_factor = get_volume_scaling_to_formula_unit(scan_natoms, element, configuration)
            
            v0_pre, _, b0_pre, b1_pre = (scan_bm_preref['min_volume']/scan_scaling_factor, scan_bm_preref['E0']/scan_scaling_factor, scan_bm_preref['bulk_modulus_ev_ang3'], scan_bm_preref['bulk_deriv'])
            delta_preref = delta(v0_pre, b0_pre, b1_pre, v0_ref_fu, b0_ref, b1_ref, 1.0, 0, 0)
            nu_preref = nu(v0_pre, b0_pre, b1_pre, v0_ref_fu, b0_ref, b1_ref, PREFACTOR, NU_PRIME_WEIGHT_B0, NU_PRIME_WEIGHT_B1)
            epsilon_preref = epsilon(v0_pre, b0_pre, b1_pre, v0_ref_fu, b0_ref, b1_ref, PREFACTOR, 0, 0)
            
            v0_refi, _, b0_refi, b1_refi = (scan_bm_refined['min_volume']/scan_scaling_factor, scan_bm_refined['E0']/scan_scaling_factor, scan_bm_refined['bulk_modulus_ev_ang3'], scan_bm_refined['bulk_deriv'])
            delta_refined = delta(v0_refi, b0_refi, b1_refi, v0_ref_fu, b0_ref, b1_ref, 1.0, 0, 0)
            nu_refined = nu(v0_refi, b0_refi, b1_refi, v0_ref_fu, b0_ref, b1_ref, PREFACTOR, NU_PRIME_WEIGHT_B0, NU_PRIME_WEIGHT_B1)
            epsilon_refined = epsilon(v0_refi, b0_refi, b1_refi, v0_ref_fu, b0_ref, b1_ref, PREFACTOR, 0, 0)

            if delta_refined > 1e-9: scores_delta.append(delta_preref / delta_refined)
            if nu_refined > 1e-9: scores_nu.append(nu_preref / nu_refined)
            if epsilon_refined > 1e-9: scores_epsilon.append(epsilon_preref / epsilon_refined)
        
        if scores_delta and scores_nu and scores_epsilon:
            avg_score_delta = np.exp(np.mean(np.log(scores_delta)))
            avg_score_nu = np.exp(np.mean(np.log(scores_nu)))
            avg_score_epsilon = np.exp(np.mean(np.log(scores_epsilon)))
            
            analysis_data.append({
                'material': material_key,
                'electronic_category': electronic_category,
                'score_delta': avg_score_delta, 
                'score_nu': avg_score_nu, 
                'score_epsilon': avg_score_epsilon,
            })

    if not analysis_data:
        sys.exit("No valid data to plot after analysis.")
        
    
    
    print("\nGenerating boxplots of refinement score vs. electronic character (G/M/U)...")
    
    for score_key, metric_symbol in [('score_delta', r'$\Delta$'), ('score_nu', r'$\nu$'), ('score_epsilon', r'$\varepsilon$')]:
        fig, ax = pl.subplots(figsize=(3.3, 3.0)) 
        
        data_by_category = defaultdict(list)
        for item in analysis_data:
           
            if item['electronic_category'] != 'Unclassified':
                data_by_category[item['electronic_category']].append(item[score_key])
        
        
        categories = ["Gap > 0", "Metallic"]
        data_to_plot = [data_by_category[c] for c in categories if c in data_by_category]
        plot_labels = [c for c in categories if c in data_by_category]

        ax.boxplot(data_to_plot, medianprops={'color': 'black', 'linestyle': '--', 'lw': 1.5}, labels=plot_labels, showfliers=False)
        
        ax.set_ylabel(f'Improvement Score S({metric_symbol})')
        ax.set_xlabel('Electronic Character at 0 K')
        ax.grid(True, axis='y', linestyle=':')
        pl.setp(ax.get_xticklabels(), rotation=0, ha="center")
        fig.tight_layout()
        
        plot_filename = os.path.join(output_dir, f"boxplot_vs_electronic_category_GMU_{score_key}.pdf")
        pl.savefig(plot_filename, dpi=300)
        pl.close(fig)

    print(f"\nEffectiveness analysis plots generated in '{output_dir}'.")
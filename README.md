# DFTK Refinement
Thoses scripts are an adapted version of the verification script for the Aiida Common Workflows subproject (https://github.com/aiidateam/aiida-common-workflows). The original works comes from:

>  [E. Bosoni et al., *How to verify the precision of density-functional-theory implementations via reproducible and universal workflows*, **Nat. Rev. Phys. 6**, 45 (2024)](https://doi.org/10.1038/s42254-023-00655-3)

## Content of the folders

- `0-preliminary-do-not-run`: ignore this folder
  
- `1-preliminary`: scripts to import the group with the initial structures, and to create a subgroup containing only the nodes you actually want to run.

- `2-submit`: scripts to submit (in batches) the common EOS workflows for all systems
  - `launch_calculations_dftk_sweep.py`: The main script for launching calculations while factoring Ecut from Ecut low to Ecut high.
  - `launch_calculations_dftk_high_ecut.py`: Script for launching the reference baseline
- `3-analyze`: scripts to analyze the results and create some output (plots, JSON with data and fitting parameters, ...)

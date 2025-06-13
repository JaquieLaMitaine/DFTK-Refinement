#!/usr/bin/env runaiida
import sys
import time
from aiida.plugins import DataFactory, WorkflowFactory
from aiida import orm
from aiida.engine import submit
from aiida_common_workflows.common import ElectronicType, RelaxType, SpinType
from aiida_common_workflows.plugins import get_entry_point_name_from_class
from aiida_common_workflows.plugins import load_workflow_entry_point
from aiida_submission_controller import FromGroupSubmissionController

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DRY_RUN = False  # False to really submit
NUM_PROCS = 6
MAX_CONCURRENT = 4
PLUGIN_NAME = "dftk"
CODE_LABEL = "DFTK_11_new3@HelvetiosHPC"
REFERENCE_PROTOCOL_FOR_ECUT = 'refinement' 
BASE_EOS_PROTOCOL = 'moderate'
# -----------------------------------------------------------------------------
# SUBMISSION CONTROLLER
# -----------------------------------------------------------------------------
class HighEcutEosSubmissionController(FromGroupSubmissionController):
    """Soumet des EOS WorkChains en utilisant un Ecut élevé fixe."""

    def __init__(self, code_label: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._code = orm.load_code(code_label)
        self._process_class = WorkflowFactory("common_workflows.eos")
        self._dftk_relax_generator_cls = load_workflow_entry_point("relax", "dftk")
        self._dftk_relax_generator = self._dftk_relax_generator_cls.get_input_generator()


    def get_extra_unique_keys(self):
        return ["element", "configuration"]

    def get_inputs_and_processclass_from_extras(self, extras_values):
        structure = self.get_parent_node_from_extras(extras_values)
        sub_relax_cls_name = get_entry_point_name_from_class(self._dftk_relax_generator_cls).name

        # Determine the HIGH Ecut value
        # Use the REFERENCE_PROTOCOL_FOR_ECUT to get the pseudo_family and high stringency settings
        protocol_details_for_ecut = self._dftk_relax_generator.get_protocol(REFERENCE_PROTOCOL_FOR_ECUT)
        pseudo_family_label = protocol_details_for_ecut['pseudo_family']
        
        try:
            pseudo_family = orm.Group.collection.get(label=pseudo_family_label)
        except orm.exc.NotExistent:
            print(f"ERROR: Pseudo family '{pseudo_family_label}' not found for structure {structure.pk}!")
            raise ValueError(f"Pseudo family '{pseudo_family_label}' not found.")


        # Get the recommended high cutoff (stringency='high')
        # This is what the 'refinement' protocol would use for its refinement step.
        recommended_high_ecut_wfc_Ha, _ = pseudo_family.get_recommended_cutoffs(
            structure=structure, stringency="high", unit='Eh' # DFTK takes Ecut in Hartree
        )
        # The refinement step in DftkCommonRelaxInputGenerator uses 2.0 * this value
        target_high_ecut_Ha = 2.0*recommended_high_ecut_wfc_Ha
        
        element, configuration = extras_values
        print(f"  Structure {element}-{configuration} (PK: {structure.pk}): Determined target_high_ecut_Ha = {target_high_ecut_Ha:.4f} Ha")


        # Prepare engines dictionary
        engines = {}
        # The DftkCommonRelaxInputGenerator.define() defines engines.relax
        for engine_name in self._dftk_relax_generator.spec().inputs['engines'].keys():
            engines[engine_name] = {
                'code': CODE_LABEL,
                'options': {
                    'resources': {
                        'num_machines': 1,
                        'num_mpiprocs_per_machine': NUM_PROCS,
                    },
                    'max_wallclock_seconds': 3600,
                }
            }
        
        # Inputs for the EquationOfStateWorkChain
        inputs = {
            'structure': structure,
            'generator_inputs': {
                'engines': engines,
                'protocol': BASE_EOS_PROTOCOL, 
                'relax_type': RelaxType.NONE,
                'electronic_type': ElectronicType.UNKNOWN,
                'spin_type': SpinType.NONE,
            },
            'sub_process_class': sub_relax_cls_name,
            'sub_process': { # Overrides for the DftkCommonRelaxWorkChain
                # The DftkCommonRelaxWorkChain will pass these down.
                # We need to target the parameters for the actual DFTK calculation
                'dftk': {
                    'parameters': orm.Dict(dict={
                        'basis_kwargs': { # This is where Ecut is set for DFTK
                            'Ecut': target_high_ecut_Ha
                        },
                    })
                }
            }
        }

        builder = self._process_class.get_builder()
        builder._update(inputs)
        builder.metadata.label = (
            f"eos_highEcut_{element}_{configuration}_EcutHa{target_high_ecut_Ha:.2f}"
        )
        return builder

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        SET_NAME = sys.argv[1]
    except IndexError:
        print("Pass as parameter the set name, e.g. oxides-verification-PBE-v1")
        sys.exit(1)

    structures_group_label = f"acwf-verification/{SET_NAME}/structures/{PLUGIN_NAME}"
    # New group for these high-Ecut baseline workflows
    workflows_group_label = f'acwf-verification/{SET_NAME}/workflows/{PLUGIN_NAME}/2high_ecut_baseline'
    
    orm.Group.collection.get_or_create(label=workflows_group_label)
    print(f"\n=== Submitting EOS calculations with fixed HIGH Ecut ===")
    print(f"    Target workflow group: {workflows_group_label}")

    controller = HighEcutEosSubmissionController(
        parent_group_label=structures_group_label,
        code_label=CODE_LABEL,
        group_label=workflows_group_label,
        max_concurrent=MAX_CONCURRENT,
    )

    print('Already run    :', controller.num_already_run)
    print('Max concurrent :', controller.max_concurrent)
    print('Available slots:', controller.num_available_slots)
    print('Active slots   :', controller.num_active_slots)
    print('Still to run   :', controller.num_to_run)
    print()

    run_processes = controller.submit_new_batch(dry_run=DRY_RUN)
    for run_process_extras, run_process in run_processes.items():
        element, configuration = run_process_extras
        if run_process is None:
            print(f'{element}-{configuration} --> To be run')    
        else:
            print(f'{element}-{configuration} --> PK = {run_process.pk}, Label = {run_process.label}')
    print("\nAll EOS workflows with high Ecut submitted (or dry-run).")

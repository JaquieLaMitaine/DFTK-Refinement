#!/usr/bin/env runaiida
import sys
import time
from aiida.plugins import DataFactory, WorkflowFactory
#!/usr/bin/env runaiida
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
MAX_CONCURRENT = 1 # NUM_PROCS//6
PLUGIN_NAME = "dftk"
CODE_LABEL = "DFTK_11_new3@HelvetiosHPC"
ECUT_FACTORS = [0.025,0.05,0.075,0.100,0.125,0.150,0.175,0.200,0.225,0.250,0.275,0.300,0.325,0.350,0.375,0.400,0.425,0.450,0.475,0.5]  # Ecut factors to sweep
# -----------------------------------------------------------------------------
# SUBMISSION CONTROLLER
# -----------------------------------------------------------------------------
class EosSubmissionController(FromGroupSubmissionController):
    """Soumet des EOS WorkChains en balayant l'Ecut initial."""

    def __init__(self, code_label: str, ecut_factor: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._code = orm.load_code(code_label)
        self._process_class = WorkflowFactory("common_workflows.eos")
        self._ecut_factor = ecut_factor

    def get_extra_unique_keys(self):
        return ["element", "configuration"]

    def get_inputs_and_processclass_from_extras(self, extras_values):
        structure = self.get_parent_node_from_extras(extras_values)
        sub_cls = load_workflow_entry_point("relax", "dftk")
        generator = sub_cls.get_input_generator()
        sub_cls_name = get_entry_point_name_from_class(sub_cls).name

        # Determine the Ecut
        protocol = generator.get_protocol("refinement")
        pseudo_family_label = protocol["pseudo_family"]
        pseudo_family = orm.Group.collection.get(label=pseudo_family_label)
        high_ecut_wfc, _ = pseudo_family.get_recommended_cutoffs(
            structure=structure, stringency="high", unit="Eh"
        )
        low_ecut = self._ecut_factor * 2.0 * high_ecut_wfc
        print("\n LOW ECUT IS : ",low_ecut)

        engine_types = generator.spec().inputs['engines']
        engines = {}
        for engine in engine_types:
            engines[engine] = {
                'code': CODE_LABEL,
                'options': {
                    'resources': {
                        'num_machines': 1,
                        'num_mpiprocs_per_machine': NUM_PROCS,
                    },
                    'max_wallclock_seconds': 3000, 
                }
            }
        inputs = {
            'structure': structure,
            'generator_inputs': {  
                'engines': engines,
                'protocol': 'refinement',
                'relax_type': RelaxType.NONE,
                'electronic_type': ElectronicType.UNKNOWN, 
                'spin_type': SpinType.NONE,
            },
            'sub_process_class': sub_cls_name,
            'sub_process': {
                'dftk': {
                    'parameters': orm.Dict(dict={
                        'basis_kwargs': {
                            'Ecut': low_ecut
                        }
                    })
                }
            }
        }


        builder = self._process_class.get_builder()
        builder._update(inputs)
        return builder

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        SET_NAME = sys.argv[1]
    except IndexError:
        print("Pass as parameter the set name, e.g. oxides-verification-PBE-v1 or unaries-verification-PBE-v1")
        sys.exit(1)

    set_name = sys.argv[1]
    structures_group_label = f"acwf-verification/{set_name}/structures/{PLUGIN_NAME}"

    for factor in ECUT_FACTORS:
        workflows_group_label = f'acwf-verification/{set_name}/workflows/{PLUGIN_NAME}/ecut_factor_smoother_17precLAST/{factor}'
        orm.Group.collection.get_or_create(workflows_group_label)
        print("\n=== ecut_factor =", factor, "===")
        controller = EosSubmissionController(
            parent_group_label=structures_group_label,
            code_label=CODE_LABEL,
            group_label=workflows_group_label,
            max_concurrent=MAX_CONCURRENT,
            ecut_factor=factor,
        )
    
        print('Already run    :', controller.num_already_run)
        print('Max concurrent :', controller.max_concurrent)
        print('Available slots:', controller.num_available_slots)
        print('Active slots   :', controller.num_active_slots)
        print('Still to run   :', controller.num_to_run)
        print()

        run_processes = controller.submit_new_batch(dry_run=DRY_RUN)
        for run_process_extras, run_process in run_processes.items():
            if run_process is None:
                print(f'{run_process_extras} --> To be run')    
            else:
                print(f'{run_process_extras} --> PK = {run_process.pk}')
    print()


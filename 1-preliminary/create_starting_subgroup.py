#!/usr/bin/env runaiida
import sys
from aiida.plugins import DataFactory
from aiida import orm
# The plugin is supposed to be dftk here
if __name__ == "__main__":

    try:
        SET_NAME = sys.argv[1]
    except IndexError:
        print("Pass as parameter the set name, e.g. oxides-verification-PBE-v1 or unaries-verification-PBE-v1")
        sys.exit(1)

    STRUCTURES_FULL_GROUP_LABEL = f'acwf-verification/{SET_NAME}/structures'
    STRUCTURES_GROUP_LABEL = f'acwf-verification/{SET_NAME}/structures/dftk'

    group = orm.Group.objects.get(label=STRUCTURES_FULL_GROUP_LABEL)
    subgroup, _ = orm.Group.objects.get_or_create(label=STRUCTURES_GROUP_LABEL)

    VALID_PAIRS = [
    ("Ba", "XO"),   ("Be", "X2O"),  ("Be", "XO"),
    ("C",  "XO"),   ("Ca", "X2O"),  ("Ca", "XO"),
    ("Cs", "X2O"),  ("Fe", "XO2"),  ("Ge", "XO"),
    ("H",  "X2O"),  ("Hf", "XO2"),  ("Ir", "X2O3"),
    ("K",  "XO"),   ("Li", "X2O"),  ("Mg", "X2O"),
    ("Mg", "XO"),   ("Na", "X2O"),  ("Os", "XO2"),
    ("Pb", "XO"),   ("Po", "XO2"),  ("Rb", "X2O"),
    ("Rn", "XO3"),  ("Ru", "XO2"),  ("Se", "XO2"),
    ("Si", "XO"),   ("Si", "XO2"),  ("Sn", "XO"),
    ("Sr", "X2O"),  ("Sr", "XO"),   ("Te", "XO2"),
    ("Ti", "X2O"),  ("Ti", "XO2"),  ("Tl", "X2O"),
    ("W",  "XO3"),  ("Zn", "X2O"),  ("Zn", "XO"),
    ("Zr", "XO2"),
    ]
    VALID_ELEMENTS    = sorted({el for el, _ in VALID_PAIRS})
    VALID_OXIDE_TYPES = sorted({ox for _, ox in VALID_PAIRS})
    VALID_PAIRS_SET   = set(VALID_PAIRS)
    print(f"Number of valid elements: {len(VALID_ELEMENTS)}")

    qb = orm.QueryBuilder()
    qb.append(
        orm.Node,
        tag="s",
        project="*",
        filters={
            "extras.element":          {"in": VALID_ELEMENTS},
            "extras.configuration":    {"in": VALID_OXIDE_TYPES},
        },
    )
    qb.append(orm.Group, with_node="s", filters={"label": STRUCTURES_FULL_GROUP_LABEL})
    wanted = [
        s for s in qb.all(flat=True)
        if (s.extras.get("element"), s.extras.get("configuration")) in VALID_PAIRS_SET
    ]
    subgroup.clear()
    subgroup.add_nodes(wanted)

    print(f"Structures from fullverdi node extras 1234  group added to group '{STRUCTURES_GROUP_LABEL}'")
    print(f"Current group size: {len(subgroup.nodes)}")
from aiida import orm, load_profile
from collections import Counter
load_profile()
#Small script to count duplicates in a group of structures
GROUP = "acwf-verification/oxides-verification-PBE-v1/structures/dftk"
group = orm.load_group(GROUP)

# Compter les (element, configuration)
pairs = [(n.base.extras.get("element"), n.base.extras.get("configuration"))
         for n in group.nodes]
dup = Counter(pairs)

print("Duplicates :")
for (el, conf), n in dup.items():
    if n > 1:
        print(f"{el:>2s} {conf:3s}  → {n} structures")

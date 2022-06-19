import sys
sys.path.append('../../../')

import mowl
mowl.init_jvm("2g")
from mowl.datasets.build_ontology import insert_annotations, create_from_triples

root = "data/"
root_mouse = "data_mouse/"
root_human = "data_human/"

species = ["human", "mouse"]

for sp in species:
    if sp == "mouse":
        root_sp = root_mouse
    elif sp == "human":
        root_sp = root_human
    else:
        raise ValueError("Species name not recognized")

    annotations = [
        (root+"gene_annots.tsv","http://has_annotation",True),
        (root+"disease_annots.tsv", "http://has_annotation", True),
        (root_sp+f"train_assoc_data_{sp}.tsv", "http://is_associated_with", False)
    ]

    insert_annotations(root+"upheno_all_with_relations.owl",annotations,root_sp+f"train_{sp}.owl")

    create_from_triples(
        root_sp+f"valid_assoc_data_{sp}.tsv",
        root_sp+f"valid_{sp}.owl",
        relation_name="is_associated_with",
        bidirectional=False,
    )

    create_from_triples(
        root_sp+f"test_assoc_data_{sp}.tsv",
        root_sp+f"test_{sp}.owl",
        relation_name="is_associated_with",
        bidirectional=False,
    )







"""Published results of Concept2Box on the DBpedia dataset (Huang et al. 2023,
arXiv:2307.01933), transcribed from the LaTeX source of the arXiv version.

* Table 2 ("Results of KG completion task"), DBpedia columns: instance-view ($\\mathcal{G}^I$)
  and ontology-view ($\\mathcal{G}^O$) completion, MRR, H@1 and H@10 (in %);
* Table 3 ("Results of concept linking task"), DBpedia columns: MRR, Acc. (Hits@1) and
  Hit@3 (in %).

Hits are printed in % and stored as fractions; MRR as printed. The ablation table (Table 4)
prints the same Concept2Box values with a third MRR digit (instance view 0.506, ontology view
0.644, concept linking 0.874). The Recipe columns are not recorded: the dataset is not public.
The baselines of the paper (TransE, RotatE, JOIE, Box4ET, ...) are not methods of mOWL. The
paper does not say whether the metrics are filtered; they are stored under the names of the
raw metrics (``mrr``, ``hits@k``).

The data of the paper is the DBpedia data of JOIE; its statistics (Table 1: 111,762 entities,
863,643 triples) are larger than the public release's (see
:class:`mowl.datasets.builtin.joie_dbpedia.JOIEDBpediaDataset`), and the split is random and
not released: the numbers are for orientation, not an exact target.
"""

from mowl.benchmarks.spec import PublishedResult
from mowl.benchmarks.entries._refs import HUANG2023

_ROWS = [
    ("joie-dbpedia-instance-view-completion", "Table 2, DBpedia, G^I KG Completion",
     {"mrr": 0.51, "hits@1": 0.3652, "hits@10": 0.7311}),
    ("joie-dbpedia-ontology-view-completion", "Table 2, DBpedia, G^O KG Completion",
     {"mrr": 0.65, "hits@1": 0.5682, "hits@10": 0.8301}),
    ("joie-dbpedia-concept-linking", "Table 3, DBpedia",
     {"mrr": 0.87, "hits@1": 0.7809, "hits@3": 0.9744}),
]

RESULTS = [PublishedResult(
    benchmark=benchmark, method="concept2box", metrics=metrics, source=HUANG2023,
    locator=locator,
    note="Concept2Box as run by its authors (bold in the table); dimensions of the instance "
         "vectors 256 and of the boxes 512 (Appendix B).")
    for benchmark, locator, metrics in _ROWS]

"""Datasets of DELE (Mashkova et al., "DELE: Deductive EL++ Embeddings for Knowledge Base
Completion"), read from the data folder of the reference implementation
(`bio-ontology-research-group/DELE <https://github.com/bio-ontology-research-group/DELE>`_,
commit ``5974c7d``).
"""

import os

from mowl.datasets.base import PathDataset, RemoteDataset, default_data_root, OWLClasses

DELE_COMMIT = "5974c7de910144993eb9ffa00ef6ceca297b5626"
DELE_BASE_URL = ("https://raw.githubusercontent.com/bio-ontology-research-group/DELE/"
                 f"{DELE_COMMIT}/data/")

#: Files of every task: (training, validation, testing), relative to ``data/``.
DELE_FILES = {
    "foodon": ("food_ontology/train_subsumption.owl", "food_ontology/valid_subsumption.owl",
               "food_ontology/test_subsumption.owl"),
    "galen": ("galen_ontology/train_subsumption.owl", "galen_ontology/valid_subsumption.owl",
              "galen_ontology/test_subsumption.owl"),
    "yeast_ppi": ("go_string/train_yeast.owl", "go_string/valid_yeast.owl",
                  "go_string/test_yeast.owl"),
    "yeast_function": ("go_string/train_yeast_hf.owl", "go_string/valid_yeast_hf.owl",
                       "go_string/test_yeast_hf.owl"),
}


def _download(url, path):
    partial = f"{path}.{os.getpid()}.part"
    try:
        with RemoteDataset._session() as session:
            with session.get(url, stream=True, timeout=(10, 60)) as req:
                req.raise_for_status()
                with open(partial, "wb") as w:
                    for chunk in req.iter_content(chunk_size=1 << 16):
                        w.write(chunk)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    os.replace(partial, path)


class DELEDataset(PathDataset):
    """Training, validation and test ontologies of one DELE experiment (Section 6.1).

    * ``"foodon"``: subsumption prediction on the Food Ontology (Table 5); the EL
      fragment of the OWL2Vec* case-study split, with axioms entailed by the training
      part removed from validation and test (5,752 test axioms; this differs from
      :class:`~mowl.datasets.builtin.FoodOnSubsumptionDataset`, which has 5,957).
    * ``"galen"``: subsumption prediction on GALEN, random 90:5:5 split (Table 6;
      667 test axioms).
    * ``"yeast_ppi"``: PPI prediction (Table 4). The test and validation files are
      byte-identical to those of :class:`~mowl.datasets.builtin.PPIYeastDataset`.
    * ``"yeast_function"``: protein function prediction,
      :math:`\\{P\\} \\sqsubseteq \\exists \\mathit{has\\_function}.\\{GO\\}` (Table 3;
      1,530 test axioms).

    The DELE release contains no human data. To apply the DELE protocol to other data,
    pass ``paths=(training, validation, testing)``.

    :param task: One of :data:`DELE_FILES`.
    :param data_root: Cache directory.
    :param paths: Local ontology files to use instead of the release.
    """

    def __init__(self, task="foodon", data_root=None, paths=None):
        self.task = task
        if paths is None:
            if task not in DELE_FILES:
                raise ValueError(
                    f"No DELE release data for task {task!r} (available: {sorted(DELE_FILES)}). "
                    f"Pass paths=(training, validation, testing) to use your own files.")
            root = default_data_root() if data_root is None else data_root
            target = os.path.join(root, f"dele_{task}")
            os.makedirs(target, exist_ok=True)
            paths = []
            for rel in DELE_FILES[task]:
                path = os.path.join(target, os.path.basename(rel))
                if not os.path.exists(path):
                    _download(DELE_BASE_URL + rel, path)
                paths.append(path)
        super().__init__(*[str(p) if p is not None else None for p in paths])

    @property
    def evaluation_classes(self):
        """Heads and tails of the evaluation: proteins (``http://4932.*`` or
        ``http://9606.*``) for PPI; proteins and GO classes for function prediction; all
        classes for subsumption prediction."""
        if self._evaluation_classes is None:
            names = self.classes.as_dict
            prots = [c for n, c in names.items() if n.startswith(("http://4932.", "http://9606."))]
            if self.task.endswith("_ppi"):
                self._evaluation_classes = OWLClasses(prots), OWLClasses(prots)
            elif self.task.endswith("_function"):
                gos = [c for n, c in names.items()
                       if n.startswith("http://purl.obolibrary.org/obo/GO_")]
                self._evaluation_classes = OWLClasses(prots), OWLClasses(gos)
            else:
                self._evaluation_classes = self.classes, self.classes
        return self._evaluation_classes

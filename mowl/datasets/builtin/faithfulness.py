"""Datasets of the faithfulness benchmark of Mashkova & Hoehndorf, "A hierarchy of
faithfulness criteria for knowledge base completion" (arXiv:2609.27863, 2026).

The data are those released with the paper's reference implementation,
`bio-ontology-research-group/kbc <https://github.com/bio-ontology-research-group/kbc>`_
(MIT licence), pinned to commit :data:`KBC_COMMIT`. For every ontology and normal form
the release has its own split (directory ``data/<ontology>/<gci>_datasets``):

* ``train.owl`` -- training ontology (normalised :math:`\\mathcal{EL}`; the test axioms of
  this normal form removed), ``train_roles.owl`` (with role inclusions and chains),
  ``val.owl`` -- validation axioms,
* ``test_ent.owl`` (``test_ent_1000.owl`` for GCI1) -- :math:`T_+`, entailed axioms,
* ``test_neg.owl`` -- :math:`T_-`, contradictory axioms (not released for the GO-plus
  slice's GCI1, GCI2 and GCI3 splits; the paper reports only :math:`\\rho_P`, MRR and
  Hits@10 there),
* ``unknown_1000.owl`` -- :math:`T_?`, undetermined axioms,

and ``pvalues/<gci>_<pizza|go>_pvalues.json`` holds the relative model counts
:math:`P(\\alpha \\mid \\mathcal{O})` of :math:`T_?` computed by the authors with Ganak
(:math:`|\\Delta| = 2`), used for :math:`\\rho_P`.

The files are downloaded individually from ``raw.githubusercontent.com`` into
``<data_root>/kbc_<commit>/`` on first use; pass ``root`` to use a local clone instead.

Observations on the released data (they are kept as released): in the Pizza GCI0 split
22 of the 936 axioms of ``test_neg.owl`` are of the form :math:`C \\sqsubseteq \\top`
(entailed, not contradictory); the contradictory sets were generated from the complete
normalised ontology, not from ``train.owl``; and the contradiction generator of the
release is rule based, so some axioms of ``unknown_1000.owl`` are contradictory under
the same non-emptiness assumption (see
:class:`mowl.evaluation.faithfulness.LogicalStatusOracle`).
"""

import json
import os

from ..base import PathDataset, RemoteDataset, default_data_root

#: Commit of bio-ontology-research-group/kbc the files are taken from.
KBC_COMMIT = "905ec7205e5640d280c2c2dec6be0aba2e8f7aad"
KBC_RAW_URL = ("https://raw.githubusercontent.com/bio-ontology-research-group/kbc/"
               f"{KBC_COMMIT}/")

NORMAL_FORMS = ("gci0", "gci1", "gci2", "gci3", "gci1_bot")


class FaithfulnessDataset(PathDataset):
    """One normal-form split of the faithfulness benchmark of [mashkova2026]_.

    The training ontology is ``train.owl``, the validation ontology ``val.owl`` and the
    testing ontology the entailed set ``test_ent.owl``. :meth:`faithfulness_test_sets`
    returns :math:`T_+, T_-, T_?` and the published relative model counts.

    :param normal_form: ``"gci0"``, ``"gci1"``, ``"gci2"``, ``"gci3"`` or ``"gci1_bot"``.
    :type normal_form: str
    :param data_root: Download cache (default :func:`mowl.datasets.base.default_data_root`).
    :type data_root: str, optional
    :param root: A local copy of the release's ``data`` directory's parent (the repository
        root, containing ``data/`` and ``pvalues/``). No download if given.
    :type root: str, optional
    """

    #: Directory of the ontology under ``data/``.
    ontology_dir = None
    #: Tag of the ontology in the p-value file names.
    pvalues_tag = None
    #: Normal forms that have a released contradictory set.
    has_contradictions = NORMAL_FORMS

    def __init__(self, normal_form="gci0", data_root=None, root=None):
        if normal_form not in NORMAL_FORMS:
            raise ValueError(f"normal_form must be one of {NORMAL_FORMS}")
        self.normal_form = normal_form
        if root is None:
            data_root = default_data_root() if data_root is None else data_root
            root = os.path.join(data_root, f"kbc_{KBC_COMMIT[:12]}")
            download = True
        else:
            download = False
        self.root = root
        split = f"data/{self.ontology_dir}/{normal_form}_datasets"
        ent = "test_ent_1000.owl" if normal_form == "gci1" else "test_ent.owl"
        files = {"train": f"{split}/train.owl", "train_roles": f"{split}/train_roles.owl",
                 "valid": f"{split}/val.owl", "entailed": f"{split}/{ent}",
                 "undetermined": f"{split}/unknown_1000.owl",
                 "pvalues": f"pvalues/{normal_form}_{self.pvalues_tag}_pvalues.json"}
        if normal_form in self.has_contradictions:
            files["contradictory"] = f"{split}/test_neg.owl"
        paths = {k: os.path.join(root, v) for k, v in files.items()}
        if download:
            for k, rel in files.items():
                self._fetch(rel, paths[k])
        self.train_roles_path = paths["train_roles"]
        self.entailed_path = paths["entailed"]
        self.contradictory_path = paths.get("contradictory")
        self.undetermined_path = paths["undetermined"]
        self.pvalues_path = paths["pvalues"]
        self._test_sets = None
        super().__init__(paths["train"], paths["valid"], paths["entailed"])

    @staticmethod
    def _fetch(rel, path):
        if os.path.exists(path):
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        partial = f"{path}.{os.getpid()}.part"
        try:
            with RemoteDataset._session() as session:
                with session.get(KBC_RAW_URL + rel, timeout=(30, 300)) as req:
                    req.raise_for_status()
                    with open(partial, "wb") as f:
                        f.write(req.content)
        except BaseException:
            if os.path.exists(partial):
                os.remove(partial)
            raise
        os.replace(partial, path)

    def _load_owl(self, path):
        if path is None:
            return None
        from java.io import File
        manager = self.ontology.getOWLOntologyManager()
        return manager.loadOntologyFromOntologyDocument(File(path))

    def load_p_values(self):
        """Published :math:`P(\\alpha \\mid \\mathcal{O})` of :math:`T_?`, keyed by axiom
        tuple (the conventions of :mod:`mowl.evaluation.faithfulness`)."""
        with open(self.pvalues_path) as f:
            return {tuple(k): float(v) for k, v in json.load(f)}

    def faithfulness_test_sets(self):
        """``{normal_form: FaithfulnessTestSet}`` with the released :math:`T_+, T_-, T_?`
        of this split and the published p-values."""
        if self._test_sets is None:
            from mowl.evaluation.faithfulness import FaithfulnessTestSet
            ts = FaithfulnessTestSet.from_owl(
                self.normal_form, self._load_owl(self.entailed_path),
                self._load_owl(self.contradictory_path), self._load_owl(self.undetermined_path),
                self.load_p_values())
            self._test_sets = {self.normal_form: ts}
        return self._test_sets

    def faithfulness_candidates(self):
        """Candidate classes of the ranked slot (MRR, Hits@k, Adm@k): all classes of the
        training ontology except ``owl:Thing`` and ``owl:Nothing``."""
        from mowl.geometric.normal_forms import BOTTOM, TOP
        return sorted(c for c in self.classes.as_str if c not in (TOP, BOTTOM))


class PizzaFaithfulnessDataset(FaithfulnessDataset):
    """The Pizza fragment of [mashkova2026]_: an :math:`\\mathcal{EL}` version of the Pizza
    ontology, normalised, with 106 classes and 4 roles (Appendix F, Table 2). Every normal
    form has a contradictory set."""

    ontology_dir = "pizza"
    pvalues_tag = "pizza"


class GOPlusFaithfulnessDataset(FaithfulnessDataset):
    """The GO-plus slice of [mashkova2026]_: a connected slice of the Gene Ontology
    (go-plus) with 653 classes and 6 roles (Appendix F, Table 2). Contradictory sets are
    released for GCI0 and GCI1-bot only, so only :math:`\\rho_P`, MRR and Hits@k are
    defined for the other normal forms."""

    ontology_dir = "GO_plus"
    pvalues_tag = "go"
    has_contradictions = ("gci0", "gci1_bot")

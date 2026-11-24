"""Semantic properties of a knowledge base embedding, after Bourgaux, Guimarães, Koudijs,
Lacerda & Ozaki, "Knowledge Base Embeddings: Semantics and Theoretical Properties", KR 2024
([BGKLO24]).

Given a *satisfaction oracle* ``satisfies(axiom) -> bool`` telling whether an embedding
:math:`E` makes a normal-form axiom true (:math:`E \\models_M \\alpha`), and a
:class:`~mowl.evaluation.faithfulness.LogicalStatusOracle` for the ontology
:math:`\\mathcal{K}`, :func:`kb_embedding_properties` checks on a finite candidate set of
concept-name axioms:

* **model** (soundness on the given axioms): :math:`E` satisfies every axiom of
  :math:`\\mathcal{K}`, i.e. :math:`E` is an :math:`M`-model (the premise of
  [BGKLO24], Property 1 and Definitions 4--6);
* **entailment closure** ([BGKLO24], Definition 4, TBox-entailed): every candidate
  entailed by :math:`\\mathcal{K}` is satisfied by :math:`E`;
* **weak TBox faithfulness** (Definition 5): every candidate satisfied by :math:`E` is
  consistent with :math:`\\mathcal{K}` (not contradictory);
* **strong TBox faithfulness** (Definition 6): every candidate satisfied by :math:`E` is
  entailed by :math:`\\mathcal{K}`.

The definitions quantify over *all* TBox axioms of the language; here they are checked on
the candidates given, so a ``True`` is a statement about those candidates only, while a
``False`` is a genuine counterexample. "Consistent with :math:`\\mathcal{K}`" in
Definition 5 is classical consistency, which corresponds to
``LogicalStatusOracle(witnesses="none")``; with the default ``"paper"`` oracle the
property checked is stronger (the axiom must not force a concept assumed non-empty to be
empty).

Relation to the hierarchy of :mod:`mowl.evaluation.faithfulness`: thresholding a score
function :math:`f` at :math:`\\theta` gives a satisfaction oracle
:math:`f(\\alpha) \\ge \\theta`. Discrimination (L0) says some threshold accepts all of
:math:`T_+` and none of :math:`T_-`; logical admissibility (L1) says some threshold gives
an entailment-closed (on :math:`T_+`) and weakly faithful oracle that accepts all of
:math:`T_?`; MLF (L2) says moreover that a higher threshold gives an entailment-closed and
strongly faithful one.
"""

from collections import defaultdict

from mowl.evaluation.faithfulness import CONTRADICTORY, ENTAILED, to_normal_form_axiom
from mowl.evaluation.regime import EvaluationRegime


def kb_embedding_properties(satisfies, oracle, candidates, axioms=None):
    """Check the properties of [BGKLO24] for one embedding.

    :param satisfies: ``satisfies(axiom) -> bool`` on :mod:`mowl.geometric.normal_forms`
        axioms (for example :meth:`mowl.geometric.ModelChecker.satisfies`).
    :param oracle: Logical status of candidates w.r.t. the ontology.
    :type oracle: :class:`~mowl.evaluation.faithfulness.LogicalStatusOracle`
    :param candidates: ``{normal_form: [axiom tuple, ...]}``.
    :param axioms: Axioms of the ontology to check for model-hood (default
        ``oracle.axioms``).
    :return: ``{"model": {...}, normal_form: {...}}``; per normal form the counts
        ``n_entailed``, ``n_entailed_satisfied``, ``n_satisfied``,
        ``n_satisfied_contradictory``, ``n_satisfied_not_entailed`` and the booleans
        ``entailment_closed``, ``weakly_faithful``, ``strongly_faithful``.
    """
    axioms = oracle.axioms if axioms is None else list(axioms)
    violated = [ax for ax in axioms if not satisfies(ax)]
    out = {"model": {"n": len(axioms), "n_violated": len(violated),
                     "fraction": (1 - len(violated) / len(axioms)) if axioms else None,
                     "is_model": not violated}}
    for nf, cands in candidates.items():
        cands = [tuple(a) for a in cands]
        status = oracle.classify(nf, cands)
        sat = [bool(satisfies(to_normal_form_axiom(nf, a))) for a in cands]
        c = defaultdict(int)
        for s, ok in zip(status, sat):
            c["n_entailed"] += s == ENTAILED
            c["n_entailed_satisfied"] += s == ENTAILED and ok
            c["n_satisfied"] += ok
            c["n_satisfied_contradictory"] += ok and s == CONTRADICTORY
            c["n_satisfied_not_entailed"] += ok and s != ENTAILED
        res = {k: int(c[k]) for k in ("n_entailed", "n_entailed_satisfied", "n_satisfied",
                                      "n_satisfied_contradictory",
                                      "n_satisfied_not_entailed")}
        res["entailment_closed"] = res["n_entailed_satisfied"] == res["n_entailed"]
        res["weakly_faithful"] = res["n_satisfied_contradictory"] == 0
        res["strongly_faithful"] = res["n_satisfied_not_entailed"] == 0
        out[nf] = res
    return out


def loss_satisfaction(model, tol=1e-6, margin=0.0):
    """Satisfaction oracle of a trained EL embedding model: an axiom holds iff its loss
    with the margin set to ``margin`` is at most ``tol`` (the criterion of
    :func:`~mowl.evaluation.faithfulness.loss_modelness`). Supports the normal forms with
    named classes (NF1 incl. :math:`\\bot`, NF2 incl. :math:`\\bot`, NF3, NF4); role
    axioms and unknown names raise :class:`KeyError`.

    :rtype: callable
    """
    import torch as th
    from mowl.geometric.normal_forms import BOTTOM, NF1, NF2, NF3, NF4
    module = model.module
    cls, rel = model.class_index_dict, model.object_property_index_dict

    def satisfies(ax):
        if isinstance(ax, NF1):
            name, row = ("gci0_bot", [cls[ax.sub], cls[BOTTOM]]) if ax.sup == BOTTOM \
                else ("gci0", [cls[ax.sub], cls[ax.sup]])
        elif isinstance(ax, NF2):
            bot = ax.sup == BOTTOM
            row = [cls[ax.sub1], cls[ax.sub2], cls.get(ax.sup, cls[ax.sub1])]
            name = "gci1_bot" if bot else "gci1"
        elif isinstance(ax, NF3):
            name, row = "gci2", [cls[ax.sub], rel[ax.role], cls[ax.filler]]
        elif isinstance(ax, NF4):
            name = "gci3_bot" if ax.sup == BOTTOM else "gci3"
            row = [rel[ax.role], cls[ax.filler], cls.get(ax.sup, cls[ax.filler])]
        else:
            raise KeyError(f"Unsupported axiom {ax!r}")
        old = getattr(module, "margin", None)
        try:
            if old is not None:
                module.margin = margin
            with th.no_grad():
                data = th.tensor([row], dtype=th.long, device=getattr(model, "device", "cpu"))
                return float(module(data, name).sum()) <= tol
        finally:
            if old is not None:
                module.margin = old
    return satisfies


class KBEmbeddingProperties(EvaluationRegime):
    """[BGKLO24] property checks as an evaluation regime.

    Candidates are generated from the dataset's training ontology with
    :func:`~mowl.evaluation.faithfulness.generate_test_sets` (union of
    :math:`T_+, T_-, T_?`). The satisfaction oracle is built by ``satisfaction(model)``,
    by default :func:`loss_satisfaction`.

    :param satisfaction: ``satisfaction(model) -> satisfies``.
    :param witnesses: Oracle mode; ``"none"`` gives Definition 5 literally.
    :param normal_forms: Normal forms of the candidates.
    :param n_per_class: Candidates per logical status and normal form.
    :param seed: Random seed.
    """

    key = "kb-embedding-properties"
    metrics = ("is_model", "entailment_closed", "weakly_faithful", "strongly_faithful")

    def __init__(self, satisfaction=loss_satisfaction, witnesses="none",
                 normal_forms=("gci0", "gci1", "gci2", "gci3", "gci1_bot"),
                 n_per_class=200, seed=0):
        self.satisfaction = satisfaction
        self.witnesses = witnesses
        self.normal_forms = tuple(normal_forms)
        self.subsets = ("model",) + self.normal_forms
        self.n_per_class = n_per_class
        self.seed = seed

    def evaluate(self, model, dataset):
        from mowl.evaluation.faithfulness import LogicalStatusOracle, generate_test_sets
        oracle = LogicalStatusOracle.from_dataset(dataset, witnesses=self.witnesses)
        n = self.n_per_class
        sets = generate_test_sets(oracle, self.normal_forms, n, n, n, seed=self.seed)
        cands = {nf: ts.entailed + ts.contradictory + ts.undetermined
                 for nf, ts in sets.items()}
        return kb_embedding_properties(self.satisfaction(model), oracle, cands)


__all__ = ["kb_embedding_properties", "loss_satisfaction", "KBEmbeddingProperties"]

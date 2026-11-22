"""Training designs of the language-model reference codes (HiT, OnT), as named regimes.

The language-model based models (:class:`~mowl.models.hit.model.HiT`,
:class:`~mowl.models.ont.model.OnT`) fine-tune a text encoder on *verbalisations* of axioms,
not on integer-indexed axiom rows. Their loop is the generic one of
:mod:`mowl.training.loop`; the training of a method is a combination of regimes that the model
names by default (``model.default_regimes``): the minibatches of triplets
(:class:`~mowl.training.EpochBatches`), the logical losses of OnT
(:class:`~mowl.training.text.SampledLogicalLosses`), the warm-up and decay of the learning rate
(:class:`~mowl.training.text.LinearWarmupDecay`) and the selection of the best epoch
(:class:`~mowl.training.ValidationPatience`).

The classes here name the designs of the reference codes. Attaching one gives the model the
combination of its own options; they are used to record, in benchmark results and recipes,
which published design a run follows
(:attr:`~mowl.benchmarks.runner.BenchmarkResult.training_regime`), and they apply to HiT and OnT
models only. The designs and their differences from the papers are documented on the
catalogue cards ``hit-code-training`` and ``ont-code-training``
(:mod:`mowl.catalog.regimes.lm_code_designs`).
"""

from mowl.training.regime import ComposedRegime

__all__ = ["LMCodeTraining", "HiTCodeTraining", "OnTCodeTraining"]


class LMCodeTraining(ComposedRegime):
    """Base class of the regimes of the LM reference codes: the combination of regimes that
    the model names for its method (:class:`~mowl.models.hit.model.HiT` and its subclasses,
    e.g. :class:`~mowl.models.ont.model.OnT`).
    """

    key = ""

    def __init__(self):
        super().__init__()
        self.key = type(self).key

    def attach(self, model):
        from mowl.models.hit.model import HiT
        if not isinstance(model, HiT):
            raise TypeError(
                f"{type(self).__name__} is the training design of a language-model "
                f"reference code and applies to HiT/OnT models only, not to "
                f"{type(model).__name__}.")
        self.regimes = tuple(r for r in (make() for make in model.default_regimes)
                             if r is not None)
        super().attach(model)


class HiTCodeTraining(LMCodeTraining):
    """How the HiT code (KRR-Oxford/HierarchyTransformers, ``scripts/training/hit``)
    trains HiT, and how the numbers of He et al. (NeurIPS 2024) were obtained.

    * A sentence-transformers model is fully fine-tuned on pre-sampled ``(child, parent,
      negative)`` triplets of entity labels with the HiT loss (clustering + centripetal,
      Eqs. 3-5).
    * AdamW, learning rate :math:`10^{-5}`, 500 linear warm-up steps followed by linear
      decay, batch size 256, 20 epochs (the paper's Appendix B; the released
      ``config_hit.yaml`` of the OnT-era repository shows the OnT margins 3.0/0.5).
    * Negatives are static: sampled once into the released datasets (10 per positive,
      random under the closed-world assumption, or siblings first for the hard-negative
      variants).
    * Model selection after every epoch by validation F1, with the decision threshold
      and the centripetal weight :math:`\\lambda` grid-searched together
      (:math:`\\lambda \\in \\{0, 0.1, \\dots, 4.9\\}`, threshold granularity 100).

    :class:`~mowl.models.hit.model.HiT` implements this design with one declared
    difference: selection is by the ranking MRR on validation subsumptions (the way the
    OnT paper evaluates HiT), not by threshold F1.
    """

    key = "hit-code-training"


class OnTCodeTraining(LMCodeTraining):
    """How the OnT code (HuiYang1997/OnT, ``train_ont.py`` and the packaged
    ``ont.pipeline.fit``) trains OnT and its HiT baseline, and how the language-model
    numbers of Yang et al. (ISWC 2025, Tables 2-4) were obtained.

    * One epoch over the verbalised training axioms of all normal forms as ``(child,
      parent, negative)`` triplets, batch size 64; one negative per axiom: the release
      stores 10 sampled negatives per axiom, and the loader keeps only the first
      (``load.py``: ``neg_list[:1]``). Negatives are uniform random concepts, *without*
      the closed-world check of HiT's own sampler; for a parent :math:`\\exists r.D`
      the negative is :math:`\\exists r.X` with the same role and a random :math:`X`.
    * Every step adds :math:`L_r` (Eq. 6) and :math:`L_\\sqcap` (Eq. 7), each on a random
      sample of the size of the batch drawn from all existential restrictions resp.
      conjunctions of the ontology, with the batch's negatives re-used in a random
      permutation; losses are means over the batch.
    * AdamW, learning rate :math:`10^{-5}`, 500 warm-up steps with linear decay, seed
      8888; after the epoch the centripetal weight is chosen on validation NF1
      subsumptions by MRR and the checkpoint with the best validation MRR is kept
      (``metric_for_best_model="MRR"``, ``load_best_model_at_end=True``).
    * The HiT baseline of the paper is trained by the same pipeline restricted to the
      hierarchy loss on the atomic (NF1) axioms; OnT(w/o r) is the same without the
      logical losses.

    Differences from the paper's text, found in the code: the paper states
    :math:`\\lambda \\in \\{0, 0.1, \\dots, 1\\}` (Section 5.1), but both the paper-time
    script and the released package search :math:`\\{0.0, \\dots, 1.9\\}`
    (``range(20) / 10``); and the centripetal loss of Eq. 6 applies to both directions
    of the :math:`\\exists r.D` alignment.
    """

    key = "ont-code-training"

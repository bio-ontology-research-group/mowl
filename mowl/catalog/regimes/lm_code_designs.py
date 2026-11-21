"""Regime cards: the training designs of the language-model reference codes of HiT
(KRR-Oxford/HierarchyTransformers) and OnT (HuiYang1997/OnT), found by comparing the
codes with the papers.

The designs are descriptive regimes (:mod:`mowl.training.lm_reference`): the
language-model models implement their reference training loop themselves, so the
regimes name and document the design rather than composing hooks.
"""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import HE2024HIT, YANG2025ONT

REGIMES = (
    RegimeCard(
        key="hit-code-training",
        assumption="closed world over stated axioms",
        name="Training design of the HiT code",
        kind=RegimeKind.TRAINING,
        summary=(
            "Full fine-tuning of a sentence-transformer (mean pooling, no normalisation) "
            "on pre-sampled (child, parent, negative) triplets of entity labels with the "
            "HiT loss (hyperbolic clustering + centripetal): AdamW, learning rate 1e-5, "
            "500 warm-up steps with linear decay, batch size 256, 20 epochs, and model "
            "selection after every epoch by validation F1, with the decision threshold "
            "and the centripetal weight λ grid-searched together (λ ∈ {0, 0.1, ..., "
            "4.9}, threshold granularity 100)."),
        components=("positives", "negative sampling", "loss", "schedule",
                    "model selection"),
        applies_to="mowl.models.hit.HiT (and subclasses)",
        references=(HE2024HIT,),
        standalone=False,
        introduced_with=("hit",),
        used_by=("hit",),
        implementation="mowl.training.lm_reference.HiTCodeTraining",
        limitations=(
            "The regime is descriptive: HiT's training loop implements the design and "
            "consults no hooks, so attaching the regime records it but does not change "
            "training.",
        ),
        notes=(
            "From scripts/training/hit/{training_hit.py, config_hit.yaml}, "
            "src/hierarchy_transformers/evaluation/hit_eval.py and Appendix B of the "
            "paper (margins α = 5.0, β = 0.1; α = 3.0 for all-mpnet-base-v2 with hard "
            "negatives; Quadro RTX 8000). The released datasets "
            "(huggingface.co/Hierarchy-Transformers) carry static negatives, 10 per "
            "positive: random under the closed-world assumption, or siblings first in "
            "the hard-negative variants ('sibling-hard-negatives'). The current "
            "repository's config_hit.yaml shows the OnT margins 3.0/0.5, not the "
            "paper's. mOWL's HiT implements this design with two declared differences: "
            "negatives are sampled by mOWL when training starts (same closed-world "
            "rule) instead of being read from a released file, and selection is by the "
            "ranking MRR on validation subsumptions with λ ∈ {0, ..., 1.0} (the way "
            "yang2025ont evaluates HiT), not by threshold F1 ('hierarchy-threshold-f1'). "
            "The HiT rows of yang2025ont's Tables 2-3 were NOT obtained with this "
            "design but with 'ont-code-training' restricted to NF1."),
    ),
    RegimeCard(
        key="ont-code-training",
        assumption="closed world over stated axioms",
        name="Training design of the OnT code",
        kind=RegimeKind.TRAINING,
        summary=(
            "Full fine-tuning of a sentence-transformer on the verbalised training "
            "axioms of all normal forms: one epoch, batch size 64, one negative per "
            "axiom (a uniform random concept, or ∃r.X with the same role for a parent "
            "∃r.D; sampled once, without a closed-world check), AdamW with learning "
            "rate 1e-5 and 500 warm-up steps (linear decay), seed 8888. Every step adds "
            "the role loss L_r (Eq. 6) and the conjunction loss L_⊓ (Eq. 7), each on a "
            "random sample of the batch's size drawn from all existential restrictions "
            "resp. conjunctions of the ontology, re-using the batch's negatives in a "
            "random permutation. After every epoch (one in the paper's setting) λ is "
            "chosen on validation NF1 subsumptions by MRR and the checkpoint with the "
            "best validation MRR is kept."),
        components=("positives", "negative sampling", "batching", "loss", "schedule",
                    "model selection"),
        applies_to="mowl.models.ont.OnT; restricted to the hierarchy loss, "
                   "mowl.models.hit.HiT (the HiT baseline of yang2025ont)",
        references=(YANG2025ONT, HE2024HIT),
        standalone=False,
        introduced_with=("ont",),
        used_by=("ont", "hit"),
        implementation="mowl.training.lm_reference.OnTCodeTraining",
        limitations=(
            "The regime is descriptive: the models' training loops implement the design "
            "and consult no hooks, so attaching the regime records it but does not "
            "change training.",
        ),
        notes=(
            "From train_ont.py, config.yaml, ont/data/{prepare,load}.py, "
            "ont/losses/logical_loss.py, ont/evaluation/ont_eval.py (paper-time: "
            "src/hierarchy_transformers/) of HuiYang1997/OnT (commit 9a2cdc3), and "
            "Section 5.1 of the paper. Paper-vs-code differences: the paper states λ ∈ "
            "{0, 0.1, ..., 1}, but the code searches {0.0, ..., 1.9} (range(20)/10) in "
            "both the paper-time script and the release; the release stores 10 "
            "negatives per axiom but the loader keeps only the first (neg_list[:1]); "
            "and unlike HiT's own sampler the negatives are uniform draws with no "
            "closed-world check, so a 'negative' can be a true subsumer. mOWL "
            "deviations, same distribution: mOWL samples the negatives itself from the "
            "rebuilt axioms instead of reading the stored ones, and generalises Eq. 7 "
            "to n-ary conjunctions (mean over the conjuncts). The HiT baseline of "
            "yang2025ont (Tables 2-3) is this design restricted to the hierarchy loss "
            "on the atomic axioms (Section 5.1: HiT is 'limited to taxonomic "
            "structures (i.e., NF1 axioms)'); OnT(w/o r) is it without the logical "
            "losses."),
    ),
)

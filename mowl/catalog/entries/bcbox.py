"""Binary code box embeddings, GBC-Box and VBC-Box (Zhang, Boratko, Musco & McCallum, 2022)."""

from mowl.catalog.card import (Family, Guarantee, GuaranteeKind as K, MethodCard, Origin,
                               Reference, Status)
from mowl.catalog.entries import _refs as R

ZHANG2022 = Reference(
    "zhang2022bcbox",
    "Zhang, D., Boratko, M., Musco, C., McCallum, A. (2022). Modeling Transitivity and "
    "Cyclicity in Directed Graphs via Binary Code Box Embeddings. In Advances in Neural "
    "Information Processing Systems 35 (NeurIPS 2022).",
    "https://proceedings.neurips.cc/paper_files/paper/2022/hash/"
    "44a1f18afd6d5cc34d7e5c3d8a80f63b-Abstract.html")

BORATKO2021 = Reference(
    "boratko2021",
    "Boratko, M., Zhang, D., Monath, N., Vilnis, L., Clarkson, K., McCallum, A. (2021). "
    "Capacity and Bias of Learned Geometric Embeddings for Directed Graphs. In Advances in "
    "Neural Information Processing Systems 34 (NeurIPS 2021).",
    "https://papers.nips.cc/paper/2021/hash/88d25099b103efd638163ecb40a55589-Abstract.html")

DASGUPTA2020 = Reference(
    "dasgupta2020",
    "Dasgupta, S. S., Boratko, M., Zhang, D., Vilnis, L., Li, X. L., McCallum, A. (2020). "
    "Improving Local Identifiability in Probabilistic Box Embeddings. In Advances in Neural "
    "Information Processing Systems 33 (NeurIPS 2020).",
    "https://arxiv.org/abs/2010.04831")

_TESTS = "tests/nn/test_bcbox.py::"

CARD = MethodCard(
    key="bcbox",
    name="Binary Code Box (GBC-Box / VBC-Box)",
    family=Family.GEOMETRIC,
    region="union of intersections of boxes (binary codes)",
    summary=(
        "Every vertex of a directed graph is a Gumbel box; learned binary codes select "
        "subsets of dimensions in which box containment is tested. Global codes (GBC-Box) "
        "make the edge set a union, over k code words, of intersections of per-dimension "
        "interval-containment graphs; per-vertex codes (VBC-Box) compare two vertices only "
        "in the dimensions both select. Unlike plain boxes, both can model directed cycles "
        "while keeping a bias towards transitivity."),
    logic="directed graphs; in mOWL the atomic subsumption graph / projected ontology graphs",
    concepts=("Gumbel box per graph vertex (2d parameters) plus, for VBC-Box, a per-vertex "
              "code in [0, 1]^d; for GBC-Box a shared code matrix B in [0, 1]^(d x k)"),
    roles="none (edge labels of the projected graph are dropped)",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.models.bcbox.model.BinaryCodeBox",
    module="mowl.nn.graph.bcbox.module.BinaryCodeBoxModule",
    axioms=("gci0",),
    guarantees=(
        Guarantee(
            K.LIMITATION,
            "Single (Gumbel) boxes cannot model cycles: if E(u,v) - E(v,u) = g(u) - g(v) "
            "for some g, no threshold reconstructs a graph containing a chordless directed "
            "cycle on more than two vertices; G-Box satisfies this with g(u) = -log of the "
            "box volume. This applies to the 'gbox' variant and to GBC/VBC when all codes "
            "are 1.",
            "zhang2022bcbox, Theorem 3 and Corollary 4",
            condition="always",
            verified_in_mowl=(_TESTS + "test_theorem3_premise_holds_for_gbox, "
                              + _TESTS + "test_gbox_cannot_fit_a_directed_3_cycle_but_vbc_can")),
        Guarantee(
            K.EXPRESSIVITY,
            "A chordless cycle plus one reverse edge can be modelled by G-Box in 2 "
            "dimensions.",
            "zhang2022bcbox, Proposition 1 (proof: Appendix C)",
            condition="existence of parameters and threshold"),
        Guarantee(
            K.EXPRESSIVITY,
            "For any temperatures, G-Box represents every graph that hard boxes represent "
            "(Theorem 1), hence every DAG in d = O((Delta + 2) log |V|) dimensions, Delta "
            "the maximum degree (Corollary 2, via boratko2021 Proposition 3).",
            "zhang2022bcbox, Theorem 1 (proof: Appendix B) and Corollary 2",
            condition="existence of parameters and threshold"),
        Guarantee(
            K.EXPRESSIVITY,
            "GBC-Box with k >= 2 codes and any temperatures can model any directed graph: "
            "split E by a total vertex order into two DAGs, embed each with G-Box in its own "
            "dimensions and select each block with one code.",
            "zhang2022bcbox, Theorem 5",
            condition="existence of parameters and threshold",
            verified_in_mowl=_TESTS + "test_theorem5_gbc_construction_models_the_3_cycle"),
        Guarantee(
            K.EXPRESSIVITY,
            "VBC-Box can model any directed graph with any temperatures and any threshold "
            "T > 0, in d = O(|V|^2) dimensions (one dimension per vertex pair).",
            "zhang2022bcbox, Theorem 6 and Theorem 7 with Lemma 2 (Appendix D)",
            condition="existence of parameters and threshold",
            verified_in_mowl=_TESTS + "test_theorem6_vbc_construction_models_any_graph"),
        Guarantee(
            K.EXPRESSIVITY,
            "VBC-Box needs only d = O((Delta_F + 2) log |V_F| + |V_C|^2) dimensions, with "
            "V_F a minimum feedback vertex set, Delta_F the maximum degree of the graph "
            "induced on V_F and V_C the vertices of edges that leave or enter V_F.",
            "zhang2022bcbox, Theorem 8 (Appendix D)",
            condition="existence of parameters and threshold"),
        Guarantee(
            K.EXPRESSIVITY,
            "GBC-Box and VBC-Box generalise G-Box: with all codes equal to 1 the energies "
            "coincide.",
            "zhang2022bcbox, Section 4.2 ('Generalization')",
            condition="by construction",
            verified_in_mowl=_TESTS + "test_all_codes_one_reduces_to_gbox"),
    ),
    limitations=(
        "The paper models one directed graph: no roles, no conjunction, no bottom, no "
        "individuals, and no set semantics. With codes below 1 an edge is not inclusion of "
        "one region in another (GBC: 'exists a code word j such that the projection to S_j "
        "is contained'; VBC: containment tested on D_u ∩ D_v only), so the learned relation "
        "is in general not transitive and a zero-loss embedding is not a model of an "
        "ontology in the sense of the other geometric methods.",
        "'Union of intersections' (Eq. 6) refers to edge sets (a union over code words of "
        "intersections of interval graphs), not to concept regions. GBC-Box with k code "
        "words is NOT the region family 'union of k boxes' (MultiBoxEL): there a concept is "
        "the set union of its boxes and subsumption is inclusion of unions (a preorder, so "
        "no chordless cycle either); here the relation is a disjunction over index-aligned "
        "projections. No extension to conjunction or disjointness follows from the paper.",
        "Codes weaken the transitivity bias (paper, Section 4.5, item 1); the lasso weight "
        "w_r and the number d_bin of coded dimensions trade it back. VBC-Box has 50% more "
        "parameters than G-Box of equal dimension.",
        "The dimension bounds of Theorems 5-8 are far from optimal (paper, Section 4.3).",
        "In the atomic subsumption graph of an ontology every cycle entails equivalence, so "
        "the deductive closure has no chordless cycle; the extra capacity of binary codes "
        "matters for non-transitive projected relations, not for gci0 semantics.",
        "Observed in mOWL: with the hard minimum of Eq. (8) GBC-Box often stays at the "
        "G-Box optimum on a 3-cycle when k = 2; k = 4 trained reliably in our tests.",
    ),
    intersection_closed=None,
    helly=("not applicable to the edge relation (it is not region inclusion); a vertex's "
           "box in one code subspace is an axis-parallel box, Helly number 2"),
    references=(ZHANG2022, BORATKO2021, DASGUPTA2020, R.FEDERICO2026_MULTIBOX),
    code_url="https://github.com/iesl/geometric_graph_embedding",
    notes=(
        "Reimplements G-Box (Eq. 3), GBC-Box (Eq. 8), VBC-Box (Eq. 9), d_bin and the lasso "
        "w_r ||1 - B||_1 (Section 4.5), trained with the BCE objective of Eq. (10), 32 "
        "corrupted negatives per positive and self-adversarial negative weights (Section "
        "4.4). Implemented as a GraphModel over projected edges (default: "
        "TaxonomyProjector); edge labels are dropped and a projected edge src -> dst is the "
        "paper's edge (dst, src), so the box of src lies in the box of dst. Unspecified in "
        "the paper and chosen by mOWL: codes are sigmoids of free logits; self-adversarial "
        "weights are RotatE's softmax (temperature alpha, no gradient), which turns the sum "
        "over negatives of Eq. (10) into a weighted average; for VBC-Box the lasso sums over "
        "all vertices' codes; corruptions that hit a training edge or a self-loop are "
        "masked. The public reference repository contains G-Box/T-Box but not the binary "
        "code models. No extension to EL++ conjunction or disjointness is implemented: a "
        "set-theoretic reading of the codes collapses to unions of boxes (MultiBoxEL) and "
        "loses the cycle capacity, and any other reading would be invented, not taken from "
        "the paper."),
)

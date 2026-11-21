"""Masked ABox revision: the crisp evaluator, the masking, the regime and the benchmarks."""

import numpy as np
import pytest

from mowl.benchmarks import get_benchmark, list_benchmarks, published_results
from mowl.benchmarks.recipes import list_recipes
from mowl.catalog import get_regime
from mowl.datasets import Dataset
from mowl.datasets.builtin.dfalc_revision import DFALCRevisionDataset
from mowl.evaluation.abox_revision import (CrispALC, MaskedABoxRevisionRegime,
                                           crisp_transform, ideal_grounding, mask_grounding)

NS = "http://example.org/r#"


def test_crisp_transform_has_three_values():
    data = [0.0, 0.1, 0.2, 0.5, 0.8, 0.81, 1.0]
    assert crisp_transform(data, alpha=0.8).tolist() == [0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0]
    assert crisp_transform(data, alpha=0.8, unknown="false").tolist() == \
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    # the released code: above 0.5 true, everything else (0.5 included) false
    assert crisp_transform([0.49, 0.5, 0.51], 0.5, "false").tolist() == [0.0, 0.0, 1.0]
    assert crisp_transform([0.49, 0.5, 0.51], 0.5, "half").tolist() == [0.0, 0.5, 1.0]


def _adapter():
    from mowl.owlapi import OWLAPIAdapter
    return OWLAPIAdapter()


def test_crisp_alc_follows_the_goedel_semantics():
    ad = _adapter()
    a, b = ad.create_class(NS + "A"), ad.create_class(NS + "B")
    r = ad.create_object_property(NS + "r")
    # domain of three elements; r relates 0 -> 1 and 1 -> 2
    concepts = {NS + "A": np.array([0.0, 1.0, 0.5]), NS + "B": np.array([0.0, 1.0, 1.0])}
    roles = {NS + "r": np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)}
    m = CrispALC(concepts, roles, 3)
    assert m.concept(ad.create_object_some_values_from(r, a)).tolist() == [1.0, 0.5, 0.0]
    assert m.concept(ad.create_object_all_values_from(r, a)).tolist() == [1.0, 0.5, 1.0]
    assert m.concept(ad.create_complement_of(a)).tolist() == [1.0, 0.0, 0.5]
    assert m.concept(ad.create_object_union_of(a, ad.create_complement_of(a))).tolist() == \
        [1.0, 1.0, 0.5]
    assert m.satisfies(ad.create_subclass_of(a, b)) is True          # A <= B pointwise
    assert m.satisfies(ad.create_subclass_of(b, a)) is False         # B(2) = 1 > A(2) = 0.5
    assert m.satisfies(ad.create_equivalent_classes(a, b)) is False
    assert m.satisfies(ad.create_disjoint_classes(a, b)) is False
    # A and not-B are disjoint
    assert m.satisfies(ad.create_disjoint_classes(a, ad.create_complement_of(b))) is True
    assert m.satisfies(ad.create_subclass_of(
        ad.create_object_intersection_of(a, ad.create_complement_of(a)), b)) is True
    # an axiom over an unknown name is not evaluated
    assert m.satisfies(ad.create_subclass_of(a, ad.create_class(NS + "Z"))) is None
    rate, ok, n, skipped = m.success_rate([ad.create_subclass_of(a, b),
                                           ad.create_subclass_of(b, a),
                                           ad.create_subclass_of(a, ad.create_class(NS + "Z"))])
    assert (rate, ok, n, skipped) == (0.5, 1, 2, 1)


def test_ideal_grounding_and_masking():
    ci = {NS + "A": 0, NS + "B": 1, "http://www.w3.org/2002/07/owl#Thing": 2}
    ri = {NS + "r": 0}
    ii = {NS + "x": 0, NS + "y": 1, NS + "z": 2, NS + "w": 3}
    c, r = ideal_grounding([(NS + "x", NS + "A"), (NS + "q", NS + "A")],
                           [(NS + "x", NS + "r", NS + "y")], ci, ri, ii)
    assert set(c) == {NS + "A", NS + "B"}                     # Thing is a constant
    assert c[NS + "A"].tolist() == [1.0, 0.5, 0.5, 0.5] and c[NS + "B"].tolist() == [0.5] * 4
    assert r[NS + "r"][0, 1] == 1.0 and r[NS + "r"].sum() == 1.0 + 15 * 0.5
    mc, mr, nc, nr = mask_grounding(c, r, 0.5, 0.8, np.random.default_rng(0))
    assert nc == 4 and nr == 2                                # rate * 2 * 4, rate * 1 * 4
    changed = sum(int((mc[k] != c[k]).sum()) for k in c)
    assert changed <= nc and changed >= nc - 1                # a draw can equal the old value
    for k in mc:
        new = mc[k][mc[k] != c[k]]
        assert ((new > 0.2) & (new < 0.8)).all()
    _, mr2, _, nr2 = mask_grounding(c, r, 0.5, 0.8, np.random.default_rng(0), "rate")
    assert nr2 == 8                                           # rate * 1 * 4 * 4
    again = mask_grounding(c, r, 0.5, 0.8, np.random.default_rng(0))
    assert all((again[0][k] == mc[k]).all() for k in mc)      # deterministic
    with pytest.raises(ValueError):
        mask_grounding(c, r, 0.5, 0.8, np.random.default_rng(0), "bogus")


def _tiny_ontology():
    """A <= B, B <= not C, exists r.A <= D; A(x), r(y, x)."""
    from java.util import HashSet
    ad = _adapter()
    ont = ad.create_ontology(NS + "tiny")
    a, b, c, d = (ad.create_class(NS + n) for n in "ABCD")
    r = ad.create_object_property(NS + "r")
    x, y = ad.create_individual(NS + "x"), ad.create_individual(NS + "y")
    axs = HashSet()
    axs.add(ad.create_subclass_of(a, b))
    axs.add(ad.create_subclass_of(b, ad.create_complement_of(c)))
    axs.add(ad.create_subclass_of(ad.create_object_some_values_from(r, a), d))
    axs.add(ad.create_class_assertion(a, x))
    axs.add(ad.create_object_property_assertion(r, y, x))
    ad.owl_manager.addAxioms(ont, axs)
    return ont


def test_regime_revises_a_masked_grounding_into_a_model_of_the_tbox():
    from mowl.models.dfalc.model import DFALCModel
    regime = MaskedABoxRevisionRegime(mask_rate=0.5, alpha=0.8, mask_seed=3)
    dataset = regime.prepare(Dataset(_tiny_ontology()))
    assert dataset.ontology.getAxioms().size() > 0
    from org.semanticweb.owlapi.model import AxiomType
    assert dataset.ontology.getAxioms(AxiomType.CLASS_ASSERTION).size() == 0   # ABox removed
    assert dataset.ontology.getAxioms(AxiomType.OBJECT_PROPERTY_ASSERTION).size() == 0
    assert dataset.individuals is not None and len(dataset.individuals) == 2
    model = DFALCModel(dataset, learning_rate=0.05)
    regime.initialise_model(model, dataset)
    grounding = model.concept_grounding()
    # entailed: A(x), B(x), D(y); everything else is 0.5 unless masked into (0.2, 0.8)
    assert regime._initial["completed_class_assertions"] == 3
    assert regime._initial["completed_role_assertions"] == 1
    assert regime._initial["masked_concept_cells"] == 4 and regime._initial["completion"] \
        .startswith("hermit")
    assert 0.0 <= regime._initial["masked_success_rate"] <= 1.0
    named = {k: v for k, v in grounding.items() if k.startswith(NS)}
    assert all(0.0 < v < 1.0 for vals in named.values() for v in vals)
    model.train(400)
    out = regime.evaluate(model, dataset)
    assert out == {"all": {"success_rate": 1.0}}
    assert regime.info["axioms_evaluated"] == 3 and regime.info["satisfied_axioms"] == 3
    assert regime.info["domain_size"] == 2


def test_regime_refuses_a_model_without_a_grounding():
    regime = MaskedABoxRevisionRegime()
    dataset = regime.prepare(Dataset(_tiny_ontology()))

    class NoGrounding:
        pass

    with pytest.raises(TypeError, match="grounding"):
        regime.initialise_model(NoGrounding(), dataset)
    with pytest.raises(ValueError):
        MaskedABoxRevisionRegime(mask_rate=1.5)
    with pytest.raises(ValueError):
        MaskedABoxRevisionRegime(alpha=0.3)


def test_dataset_reads_the_file_of_the_release_once(tmp_path):
    import java.io
    from org.semanticweb.owlapi.formats import FunctionalSyntaxDocumentFormat
    from org.semanticweb.owlapi.model import IRI
    ad = _adapter()
    ont = _tiny_ontology()
    release = tmp_path / "release"
    release.mkdir()
    ad.owl_manager.saveOntology(ont, FunctionalSyntaxDocumentFormat(), IRI.create(
        java.io.File(str(release / "Family.owl")).toURI()))
    data = DFALCRevisionDataset("family", data_root=str(tmp_path / "root"),
                                base_url=release.as_uri() + "/")
    assert len(data.individuals) == 2 and len(data.classes) >= 4
    again = DFALCRevisionDataset("family", data_root=str(tmp_path / "root"),
                                 base_url="file:///does/not/exist/")        # cached
    assert len(again.individuals) == 2
    with pytest.raises(ValueError):
        DFALCRevisionDataset("bogus", data_root=str(tmp_path))


def test_benchmarks_published_numbers_and_recipes_register():
    benches = [b for b in list_benchmarks() if b.key.startswith("dfalc-revision-")]
    assert len(benches) == 7 * 4
    b = get_benchmark("dfalc-revision-nifdys-mask20")
    assert b.evaluation_regime == "masked-abox-revision" and b.protocol_kwargs["mask_rate"] == 0.2
    assert str(b.assumption.value) == "closed world over entailed axioms"
    assert get_regime("masked-abox-revision").implementation.endswith(
        "MaskedABoxRevisionRegime")
    rows = published_results(benchmark="dfalc-revision-nifdys-mask20", method="dfalc")
    by_loss = {r.settings["loss_type"]: r.metrics["success_rate"] for r in rows}
    assert by_loss == {"hierarchical": 0.973, "rule": 0.892}
    assert len(published_results(method="dfalc")) == 7 * 4 * 2
    assert "7.3%" in rows[0].note
    recipes = list_recipes(benchmark="dfalc-revision-sso-mask80", method="dfalc",
                           source="wu2022")
    assert {r.name for r in recipes} == {"hierarchical", "rule"}
    assert all(r.model_kwargs["learning_rate"] == 2e-4 for r in recipes)

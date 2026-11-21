"""End-to-end tests of E2R (mowl.models.e2r): axiom extraction, hand-built zero-loss
embeddings checked with set operations independent of the losses, the T3 pattern, a
short training run with ABox, and scoring."""
import numpy as np
import pytest
import torch as th
from java.util import HashSet

from mowl.datasets import Dataset
from mowl.models.e2r.model import E2R
from mowl.nn.alc.e2r.module import E2RModule
from mowl.owlapi import OWLAPIAdapter

NS = "http://mowl/e2r#"


def _family():
    ad = OWLAPIAdapter()
    ont = ad.create_ontology("http://mowl/e2r/family")
    c = {n: ad.create_class(NS + n) for n in
         ["Person", "Male", "Female", "Father", "Mother", "Parent", "Animal"]}
    r = {n: ad.create_object_property(NS + n) for n in ["hasChild", "relatedTo"]}
    i = {n: ad.create_individual(NS + n) for n in ["john", "mary", "ann", "rex"]}
    df = ad.data_factory
    axs = HashSet()
    for a, b in [("Male", "Person"), ("Female", "Person"), ("Father", "Male"),
                 ("Mother", "Female")]:
        axs.add(ad.create_subclass_of(c[a], c[b]))
    axs.add(ad.create_disjoint_classes(c["Male"], c["Female"]))
    axs.add(ad.create_disjoint_classes(c["Person"], c["Animal"]))
    axs.add(ad.create_equivalent_classes(
        c["Parent"], ad.create_object_union_of(c["Father"], c["Mother"])))
    axs.add(df.getOWLSubObjectPropertyOfAxiom(r["hasChild"], r["relatedTo"]))
    axs.add(df.getOWLObjectPropertyRangeAxiom(r["hasChild"], c["Person"]))
    axs.add(ad.create_subclass_of(
        c["Parent"], ad.create_object_some_values_from(r["hasChild"], c["Person"])))
    for ind, cl in [("john", "Father"), ("mary", "Female"), ("ann", "Mother"),
                    ("rex", "Animal")]:
        axs.add(ad.create_class_assertion(c[cl], i[ind]))
    axs.add(ad.create_object_property_assertion(r["hasChild"], i["john"], i["mary"]))
    axs.add(ad.create_object_property_assertion(r["hasChild"], i["ann"], i["mary"]))
    ad.owl_manager.addAxioms(ont, axs)
    return Dataset(ont)


def _t3():
    """A ≡ X1 ⊓ X2, B ≡ X2 ⊓ X3, C ≡ X1 ⊓ X3, A, B, C pairwise disjoint, A(a), B(b), C(c)."""
    ad = OWLAPIAdapter()
    ont = ad.create_ontology("http://mowl/e2r/t3")
    c = {n: ad.create_class(NS + n) for n in ["X1", "X2", "X3", "A", "B", "C"]}
    axs = HashSet()
    for k, (p, q) in {"A": ("X1", "X2"), "B": ("X2", "X3"), "C": ("X1", "X3")}.items():
        axs.add(ad.create_equivalent_classes(
            c[k], ad.create_object_intersection_of(c[p], c[q])))
        axs.add(ad.create_class_assertion(c[k], ad.create_individual(NS + k.lower())))
    axs.add(ad.create_disjoint_classes(c["A"], c["B"], c["C"]))
    ad.owl_manager.addAxioms(ont, axs)
    return Dataset(ont)


def _set(model, concepts=None, relations=None, entities=None):
    """Write exact parameters (dicts IRI -> real list) into the model."""
    m, d = model.module, model.embed_dim
    with th.no_grad():
        m.c_embedding.weight.zero_()
        m.r_embedding.weight.zero_()
        m.e_embedding.weight.zero_()
        for name, v in (concepts or {}).items():
            m.c_embedding.weight[model.class_index_dict[NS + name], :d] = th.tensor(v)
        for name, v in (relations or {}).items():
            m.r_embedding.weight[model.object_property_index_dict[NS + name]] = th.tensor(v)
        for name, v in (entities or {}).items():
            m.e_embedding.weight[model.individual_index_dict[NS + name], :d] = th.tensor(v)


@pytest.fixture(scope="module")
def family():
    return _family()


class TestExtraction:

    def test_rows_and_skipped(self, family):
        model = E2R(family, embed_dim=4)
        rows = {k: v.shape[0] for k, v in model.training_rows.items()}
        assert rows["class_assertion"] == 4
        assert rows["role_assertion"] == 2
        assert rows["concept_inclusion"] == 4
        assert rows["concept_disjointness"] == 2
        assert rows["concept_disjunction_2"] == 1
        assert rows["role_inclusion"] == 1
        # range(hasChild, Person): shrunk non-descendants of Person are Animal (child of
        # the top) and Parent (child of the top, a superclass of Father via the definition)
        cid = model.class_index_dict
        ks = {k for _, k in model.training_rows["range"].tolist()}
        assert ks == {cid[NS + "Animal"], cid[NS + "Parent"]}
        assert len(model.skipped_axioms) == 1
        assert "ObjectSomeValuesFrom" in str(model.skipped_axioms[0])
        assert isinstance(model.module, E2RModule)

    def test_role_assertion_order(self, family):
        model = E2R(family, embed_dim=4)
        iid, rid = model.individual_index_dict, model.object_property_index_dict
        rows = {tuple(r) for r in model.training_rows["role_assertion"].tolist()}
        assert (iid[NS + "john"], rid[NS + "hasChild"], iid[NS + "mary"]) in rows


class TestExactModels:

    def _family_model(self, family):
        # concept space R^5: axis 0 = male, 1 = female, 2 = father, 3 = animal, 4 = mother
        model = E2R(family, embed_dim=5)
        _set(model,
             concepts={"Person": [1, 1, 1, 0, 1], "Male": [1, 0, 1, 0, 0],
                       "Female": [0, 1, 0, 0, 1], "Father": [0, 0, 1, 0, 0],
                       "Mother": [0, 0, 0, 0, 1], "Parent": [0, 0, 1, 0, 1],
                       "Animal": [0, 0, 0, 1, 0]},
             relations={"hasChild": [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                        "relatedTo": [0, 0, 1, 0, 1, 1, 1, 0, 0, 0]},
             entities={"john": [0, 0, 1, 0, 0], "mary": [0, 1, 0, 0, 0],
                       "ann": [0, 0, 0, 0, 1], "rex": [0, 0, 0, 1, 0]})
        return model

    def test_zero_loss_embedding_is_a_model(self, family):
        """A hand-built zero-loss embedding satisfies every training axiom according to
        the set-based check of E2R.violations, which does not use the losses."""
        model = self._family_model(family)
        assert model.axiom_loss() == pytest.approx(0.0, abs=1e-12)
        assert model.violations() == []

    def test_violation_detected(self, family):
        model = self._family_model(family)
        with th.no_grad():  # put Father outside Male
            model.module.c_embedding.weight[model.class_index_dict[NS + "Male"], 2] = 0
        kinds = {k for k, _ in model.violations()}
        assert "concept_inclusion" in kinds

    def test_t3_exact_model_in_dimension_3(self):
        model = E2R(_t3(), embed_dim=3)
        _set(model,
             concepts={"X1": [1, 0, 1], "X2": [1, 1, 0], "X3": [0, 1, 1],
                       "A": [1, 0, 0], "B": [0, 1, 0], "C": [0, 0, 1]},
             entities={"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]})
        assert model.axiom_loss() == pytest.approx(0.0, abs=1e-12)
        assert model.violations() == []
        for name in "ABC":
            assert model.concept_indicator(NS + name).sum() == 1  # non-trivial

    def test_t3_orthogonality_regulariser_conflicts(self):
        """X1, X2, X3 are siblings (children of the top), so eq. (11) asks for X1 ⊥ X2,
        but A = X1 ⊓ X2 must be non-trivial: every model of T3 has positive Ω_orth."""
        model = E2R(_t3(), embed_dim=3)
        _set(model,
             concepts={"X1": [1, 0, 1], "X2": [1, 1, 0], "X3": [0, 1, 1],
                       "A": [1, 0, 0], "B": [0, 1, 0], "C": [0, 0, 1]})
        cid = model.class_index_dict
        pairs = {tuple(sorted(p)) for p in model.sibling_pairs.tolist()}
        x1, x2 = cid[NS + "X1"], cid[NS + "X2"]
        assert tuple(sorted((x1, x2))) in pairs
        assert float(model.module.orthogonality_loss(th.tensor([[x1, x2]])).detach().sum()) > 0

    def test_t3_has_no_model_in_dimension_2(self):
        """Exhaustive over all 0/1 indicators in d = 2: no assignment satisfies the
        TBox of T3 with A, B, C non-trivial."""
        import itertools
        model = E2R(_t3(), embed_dim=2)
        names = ["X1", "X2", "X3", "A", "B", "C"]
        found = False
        for bits in itertools.product(itertools.product([0, 1], repeat=2), repeat=6):
            if any(sum(b) == 0 for b in bits[3:]):
                continue
            _set(model, concepts=dict(zip(names, [list(b) for b in bits])))
            if not [v for v in model.violations() if v[0] != "class_assertion"]:
                found = True
                break
        assert not found


class TestTraining:

    def test_training_and_scoring(self, family):
        th.manual_seed(0)
        model = E2R(family, embed_dim=8, learning_rate=0.05, sep_gap=1.0)
        before = model.total_loss()
        history = model.train(epochs=300)
        after = model.total_loss()
        assert after < before
        assert len(history) == 300
        # Deductive membership: Person has no asserted member; the three people must
        # rank above the animal.
        ranking = model.rank_members(NS + "Person")
        assert ranking[-1] == NS + "rex"
        scores = model.score_membership(NS + "Person")
        assert scores.shape == (4,) and np.all(scores >= 0)
        rs = model.score_role_membership(
            NS + "relatedTo", [(NS + "john", NS + "mary"), (NS + "mary", NS + "john")])
        assert rs[0] < rs[1]
        assert model.score_subsumption(NS + "Father", NS + "Person") < \
            model.score_subsumption(NS + "Person", NS + "Father")

    def test_complex_expressions(self, family):
        model = E2R(family, embed_dim=4)
        ad = OWLAPIAdapter()
        f, m = ad.create_class(NS + "Father"), ad.create_class(NS + "Mother")
        _set(model, concepts={"Father": [1, 0, 0, 0], "Mother": [0, 1, 0, 0]})
        u = model.concept_indicator(ad.create_object_union_of(f, m))
        n = model.concept_indicator(ad.create_complement_of(f))
        i = model.concept_indicator(ad.create_object_intersection_of(f, m))
        assert u.tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
        assert n.tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
        assert i.sum() == 0
        with pytest.raises(NotImplementedError):
            model.concept_indicator(ad.create_object_some_values_from(
                ad.create_object_property(NS + "hasChild"), f))

    @pytest.mark.slow
    def test_t3_learned(self):
        """Without the sibling-orthogonality regulariser (which penalises every model of
        T3, see test_t3_orthogonality_regulariser_conflicts) training finds an exact
        model of T3 in d = 6 for at least one of three seeds."""
        found = []
        for seed in range(3):
            th.manual_seed(seed)
            model = E2R(_t3(), embed_dim=6, learning_rate=0.05, sep_gap=1.0, reg_orth=0)
            model.train(epochs=1500)
            found.append(model.violations() == [])
        assert any(found)

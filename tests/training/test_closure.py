"""Deductive closure (DELE, Algorithms 1 and 2) on small hand-written ontologies."""
import itertools
import os
import tempfile
from unittest import TestCase

import torch as th
from java.util import HashSet

from mowl.owlapi import OWLAPIAdapter
from mowl.owlapi.defaults import TOP, BOT
from mowl.training import DeductiveClosure

NS = "http://test.dele/"


def _ontology(build):
    ad = OWLAPIAdapter()
    C = lambda n: ad.create_class(NS + n)  # noqa: E731
    R = lambda n: ad.create_object_property(NS + n)  # noqa: E731
    ont = ad.create_ontology(NS + "ont" + str(id(build)))
    axioms = build(ad, C, R)
    ad.owl_manager.addAxioms(ont, HashSet(axioms))
    return ont


def appendix_e(ad, C, R):
    """DELE, Appendix E: the protein-function example (already normalised)."""
    hf = R("hf")
    some = ad.create_object_some_values_from
    inter = ad.create_object_intersection_of
    sub = ad.create_subclass_of
    nothing = ad.create_class(BOT)
    return [sub(inter(C("GO1"), C("GO2")), nothing),
            sub(inter(C("A"), C("B")), nothing),
            sub(some(hf, C("GO1")), C("B")),
            sub(some(hf, C("GO2")), C("A")),
            sub(C("P"), some(hf, C("GO1"))),
            sub(C("Q"), some(hf, C("GO2")))]


def roles_and_chains(ad, C, R):
    some = ad.create_object_some_values_from
    sub = ad.create_subclass_of
    df = ad.data_factory
    from java.util import ArrayList
    chain = ArrayList()
    chain.add(R("r"))
    chain.add(R("s"))
    return [sub(C("X"), some(R("r"), C("Y"))),
            sub(C("Y"), some(R("s"), C("Z"))),
            sub(C("X2"), C("X")),
            sub(C("Z"), C("Z2")),
            df.getOWLSubObjectPropertyOfAxiom(R("r"), R("rr")),
            df.getOWLSubPropertyChainOfAxiom(chain, R("t")),
            sub(some(R("rr"), C("Y")), C("W"))]


def _elk_sub(reasoner, sub, sup):
    """ELK has no isEntailed: test sub ⊑ sup through the class hierarchy."""
    if not reasoner.isSatisfiable(sub):
        return True
    if sup.isOWLThing() or sub.equals(sup):
        return True
    sups = set(reasoner.getSuperClasses(sub, False).getFlattened())
    sups |= set(reasoner.getEquivalentClasses(sub).getEntities())
    return sup in sups


class TestClosure(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ont_e = _ontology(appendix_e)
        cls.ont_r = _ontology(roles_and_chains)
        cls.closures = {
            (name, rs): DeductiveClosure.from_ontology(ont, reasoner=rs)
            for name, ont in (("e", cls.ont_e), ("r", cls.ont_r))
            for rs in ("canonical", "elk")}

    def _ids(self, cl, *names):
        out = []
        for n in names:
            iri = {"TOP": TOP, "BOT": BOT}.get(n, NS + n)
            out.append(cl.class_index_dict[iri])
        return out

    def _tails(self, cl, tails):
        return {cl._id_to_class[t] for t in tails if t < len(cl.class_index_dict)}

    def test_appendix_e_hierarchy(self):
        cl = self.closures[("e", "canonical")]
        p, q, a, b = self._ids(cl, "P", "Q", "A", "B")
        self.assertEqual(self._tails(cl, cl.superclasses(p)), {NS + "P", NS + "B", TOP})
        self.assertEqual(self._tails(cl, cl.superclasses(q)), {NS + "Q", NS + "A", TOP})
        self.assertEqual(self._tails(cl, cl.superclasses(a)), {NS + "A", TOP})

    def test_appendix_e_gci2_and_gci3(self):
        cl = self.closures[("e", "canonical")]
        hf = cl.object_property_index_dict[NS + "hf"]
        p, go1, go2, b, a = self._ids(cl, "P", "GO1", "GO2", "B", "A")
        self.assertEqual(self._tails(cl, cl.entailed_tails("gci2", p, hf)), {NS + "GO1", TOP})
        self.assertIsNone(cl.entailed_tails("gci2", cl.bot, hf))  # ⊥ ⊑ ∃hf.X
        self.assertEqual(self._tails(cl, cl._gci3_tails(hf, go1)), {NS + "B", TOP})
        self.assertEqual(self._tails(cl, cl._gci3_tails(hf, p)), {TOP})
        self.assertTrue(cl.entails("gci3", [hf, go2, a]))
        self.assertFalse(cl.entails("gci3", [hf, go2, b]))

    def test_appendix_e_gci1_rows(self):
        cl = self.closures[("e", "canonical")]
        p, q, a, b, go1, go2 = self._ids(cl, "P", "Q", "A", "B", "GO1", "GO2")
        everything = set(cl.class_index_dict)
        self.assertIsNone(cl._gci1_tails(p, q))          # {P} ⊓ {Q} ⊑ everything
        self.assertIsNone(cl._gci1_tails(p, a))          # {P} ⊓ A ⊑ everything
        self.assertIsNone(cl._gci1_tails(go1, go2))
        self.assertEqual(self._tails(cl, cl._gci1_tails(p, go1)),
                         {NS + "P", NS + "GO1", NS + "B", TOP})
        self.assertEqual(self._tails(cl, cl._gci1_tails(a, go2)), {NS + "A", NS + "GO2", TOP})
        self.assertTrue(everything)
        # GCI1-BOT and commutativity
        self.assertTrue(cl.entails("gci1_bot", [a, p, cl.bot]))
        self.assertFalse(cl.entails("gci1_bot", [a, go1, cl.bot]))

    def test_roles_and_chains(self):
        cl = self.closures[("r", "canonical")]
        rel = cl.object_property_index_dict
        r, rr, t = rel[NS + "r"], rel[NS + "rr"], rel[NS + "t"]
        x, x2, y, z2, w = self._ids(cl, "X", "X2", "Y", "Z2", "W")
        self.assertTrue(cl.entails("gci2", [x2, r, y]))      # subclass rule
        self.assertTrue(cl.entails("gci2", [x2, rr, y]))     # role hierarchy
        self.assertTrue(cl.entails("gci2", [x, t, z2]))      # chain rule + superclass
        self.assertFalse(cl.entails("gci2", [y, t, z2]))
        self.assertTrue(cl.entails("gci3", [r, y, w]))       # r ⊑ rr, ∃rr.Y ⊑ W
        self.assertTrue(cl.entails("gci0", [x, w]))          # found by the reasoner

    def test_canonical_equals_elk(self):
        """Both reasoners give the same closure for every normal form and every tuple."""
        for name in ("e", "r"):
            c1, c2 = self.closures[(name, "canonical")], self.closures[(name, "elk")]
            n, m = len(c1.class_index_dict), len(c1.object_property_index_dict)
            self.assertEqual(c1.class_index_dict, c2.class_index_dict)
            for a, b in itertools.product(range(n), repeat=2):
                self.assertEqual(c1.entails("gci0", [a, b]), c2.entails("gci0", [a, b]))
                for r in range(m):
                    self.assertEqual(c1.entails("gci2", [a, r, b]), c2.entails("gci2", [a, r, b]))
                    self.assertEqual(c1.entails("gci3", [r, a, b]), c2.entails("gci3", [r, a, b]))
                for e in range(n):
                    self.assertEqual(c1.entails("gci1", [a, b, e]), c2.entails("gci1", [a, b, e]))

    def test_closure_is_sound(self):
        """Every enumerated GCI0/GCI2 axiom is entailed (checked with ELK and a fresh
        class for the existential)."""
        from org.semanticweb.elk.owlapi import ElkReasonerFactory
        ad = OWLAPIAdapter()
        for name, ont in (("e", self.ont_e), ("r", self.ont_r)):
            cl = self.closures[(name, "canonical")]
            inv_c = cl._id_to_class
            inv_r = {v: k for k, v in cl.object_property_index_dict.items()}
            reasoner = ElkReasonerFactory().createReasoner(ont)
            for a, b in cl.axioms("gci0").tolist():
                self.assertTrue(_elk_sub(reasoner, ad.create_class(inv_c[a]),
                                         ad.create_class(inv_c[b])), (inv_c[a], inv_c[b]))
            reasoner.dispose()
            for a, r, b in cl.axioms("gci2").tolist():
                fresh = ad.create_class(NS + "Fresh")
                extra = ad.create_ontology(NS + f"sound{name}{a}{r}{b}")
                axioms = HashSet(ont.getAxioms())
                axioms.add(ad.create_subclass_of(
                    ad.create_object_some_values_from(ad.create_object_property(inv_r[r]),
                                                      ad.create_class(inv_c[b])), fresh))
                ad.owl_manager.addAxioms(extra, axioms)
                rs = ElkReasonerFactory().createReasoner(extra)
                rs.precomputeInferences()
                self.assertTrue(_elk_sub(rs, ad.create_class(inv_c[a]), fresh),
                                (inv_c[a], inv_r[r], inv_c[b]))
                rs.dispose()

    def test_contains_and_cache(self):
        cl = self.closures[("e", "canonical")]
        p, b, a = self._ids(cl, "P", "B", "A")
        mask = cl.contains("gci0", th.tensor([[p, b], [p, a]]))
        self.assertEqual(mask.tolist(), [True, False])
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/tmp")) as d:
            c1 = DeductiveClosure.from_ontology(self.ont_e, cache_dir=d)
            files = os.listdir(d)
            self.assertEqual(len(files), 1)
            c2 = DeductiveClosure.from_ontology(self.ont_e, cache_dir=d)
            self.assertEqual(c1._sups, c2._sups)

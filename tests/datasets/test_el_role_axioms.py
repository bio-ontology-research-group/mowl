import mowl
from mowl import init_jvm
init_jvm("1g")

from unittest import TestCase
from org.semanticweb.owlapi.model import IRI
from jpype import java
from mowl.owlapi import OWLAPIAdapter
from mowl.datasets.el.el_dataset import ELDataset
from mowl.ontology.normalize import extract_role_axioms, RoleInclusion, RoleChain


def create_ontology_with_role_axioms():
    """Creates an in-memory ontology with EL GCIs, a role inclusion, a transitive
    property declaration and a 2-property role chain."""
    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    factory = adapter.data_factory

    c1 = factory.getOWLClass(IRI.create("http://test/C1"))
    c2 = factory.getOWLClass(IRI.create("http://test/C2"))
    c3 = factory.getOWLClass(IRI.create("http://test/C3"))
    r1 = factory.getOWLObjectProperty(IRI.create("http://test/R1"))
    r2 = factory.getOWLObjectProperty(IRI.create("http://test/R2"))
    r3 = factory.getOWLObjectProperty(IRI.create("http://test/R3"))

    ont = manager.createOntology(IRI.create("http://test/role_axioms"))
    # gci0: C1 ⊑ C2
    ont.addAxiom(factory.getOWLSubClassOfAxiom(c1, c2))
    # gci2: C2 ⊑ ∃R1.C3
    ont.addAxiom(factory.getOWLSubClassOfAxiom(
        c2, factory.getOWLObjectSomeValuesFrom(r1, c3)))
    # role inclusion: R1 ⊑ R2
    ont.addAxiom(factory.getOWLSubObjectPropertyOfAxiom(r1, r2))
    # transitive property: R2
    ont.addAxiom(factory.getOWLTransitiveObjectPropertyAxiom(r2))
    # role chain: R2 ∘ R3 ⊑ R1
    chain = java.util.ArrayList()
    chain.add(r2)
    chain.add(r3)
    ont.addAxiom(factory.getOWLSubPropertyChainOfAxiom(chain, r1))
    return ont


class TestExtractRoleAxioms(TestCase):

    def setUp(self):
        self.ont = create_ontology_with_role_axioms()

    def test_type_check(self):
        with self.assertRaisesRegex(TypeError, "Parameter 'ontology' must be of type \
org.semanticweb.owlapi.model.OWLOntology."):
            extract_role_axioms("ontology")

    def test_extraction(self):
        role_inclusions, role_chains = extract_role_axioms(self.ont)

        self.assertEqual(len(role_inclusions), 1)
        inclusion = role_inclusions[0]
        self.assertIsInstance(inclusion, RoleInclusion)
        self.assertEqual(inclusion.sub_property, "http://test/R1")
        self.assertEqual(inclusion.super_property, "http://test/R2")
        classes, properties, individuals = inclusion.get_entities()
        self.assertEqual(classes, set())
        self.assertEqual(properties, {"http://test/R1", "http://test/R2"})
        self.assertEqual(individuals, set())

        self.assertEqual(len(role_chains), 2)
        by_super = {chain.super_property: chain for chain in role_chains}
        self.assertIsInstance(by_super["http://test/R2"], RoleChain)
        self.assertEqual(by_super["http://test/R2"].sub_chain,
                         ("http://test/R2", "http://test/R2"))
        self.assertIsInstance(by_super["http://test/R1"], RoleChain)
        self.assertEqual(by_super["http://test/R1"].sub_chain,
                         ("http://test/R2", "http://test/R3"))
        classes, properties, individuals = by_super["http://test/R1"].get_entities()
        self.assertEqual(classes, set())
        self.assertEqual(properties,
                         {"http://test/R2", "http://test/R3", "http://test/R1"})
        self.assertEqual(individuals, set())

    def test_non_atomic_role_expressions_are_ignored(self):
        adapter = OWLAPIAdapter()
        factory = adapter.data_factory
        r1 = factory.getOWLObjectProperty(IRI.create("http://test/S1"))
        r2 = factory.getOWLObjectProperty(IRI.create("http://test/S2"))
        r3 = factory.getOWLObjectProperty(IRI.create("http://test/S3"))
        ont = adapter.owl_manager.createOntology(IRI.create("http://test/inverse_roles"))
        # role inclusion with an inverse sub property: (R1 o) ⊑ R2
        ont.addAxiom(factory.getOWLSubObjectPropertyOfAxiom(factory.getOWLObjectInverseOf(r1), r2))
        # role chain with an inverse property in the chain: (R2 o) ∘ R3 ⊑ R1
        chain = java.util.ArrayList()
        chain.add(factory.getOWLObjectInverseOf(r2))
        chain.add(r3)
        ont.addAxiom(factory.getOWLSubPropertyChainOfAxiom(chain, r1))
        # an atomic chain: R3 ∘ R3 ⊑ R2
        chain2 = java.util.ArrayList()
        chain2.add(r3)
        chain2.add(r3)
        ont.addAxiom(factory.getOWLSubPropertyChainOfAxiom(chain2, r2))

        role_inclusions, role_chains = extract_role_axioms(ont)
        self.assertEqual(role_inclusions, [])
        self.assertEqual(len(role_chains), 1)
        self.assertEqual(role_chains[0].sub_chain, ("http://test/S3", "http://test/S3"))
        self.assertEqual(role_chains[0].super_property, "http://test/S2")

    def test_ontology_without_role_axioms(self):
        adapter = OWLAPIAdapter()
        factory = adapter.data_factory
        c1 = factory.getOWLClass(IRI.create("http://test/D1"))
        c2 = factory.getOWLClass(IRI.create("http://test/D2"))
        ont = adapter.owl_manager.createOntology(IRI.create("http://test/no_roles"))
        ont.addAxiom(factory.getOWLSubClassOfAxiom(c1, c2))

        role_inclusions, role_chains = extract_role_axioms(ont)
        self.assertEqual(role_inclusions, [])
        self.assertEqual(role_chains, [])


class TestEldatasetRoleAxioms(TestCase):

    def setUp(self):
        self.ont = create_ontology_with_role_axioms()

    def test_role_datasets(self):
        dataset = ELDataset(self.ont, extended=True, load_normalized=False)

        inclusion_dataset = dataset.role_inclusion_dataset
        self.assertIsNotNone(inclusion_dataset)
        self.assertEqual(len(inclusion_dataset), 1)
        data = inclusion_dataset.data
        self.assertEqual(data.shape[1], 2)

        chain_dataset = dataset.role_chain_dataset
        self.assertIsNotNone(chain_dataset)
        self.assertEqual(len(chain_dataset), 2)
        data = chain_dataset.data
        self.assertEqual(data.shape[1], 3)

        index_dict = dataset.object_property_index_dict
        # R1 (inclusion + chain), R2 (inclusion + transitive + chain), R3 (chain)
        r1 = index_dict["http://test/R1"]
        r2 = index_dict["http://test/R2"]
        r3 = index_dict["http://test/R3"]
        self.assertEqual(inclusion_dataset.data.tolist(), [[r1, r2]])
        self.assertEqual(sorted(map(tuple, chain_dataset.data.tolist())),
                         sorted([(r2, r2, r2), (r2, r3, r1)]))

    def test_get_gci_datasets_includes_role_axioms(self):
        dataset = ELDataset(self.ont, extended=True, load_normalized=False)
        datasets = dataset.get_gci_datasets()

        self.assertIn("role_inclusion", datasets)
        self.assertIn("role_chain", datasets)
        self.assertEqual(len(datasets["role_inclusion"]), 1)
        self.assertEqual(len(datasets["role_chain"]), 2)
        # The GCIs are still there
        self.assertEqual(len(datasets["gci0"]), 1)
        self.assertEqual(len(datasets["gci2"]), 1)

    def test_ontology_without_role_axioms(self):
        adapter = OWLAPIAdapter()
        factory = adapter.data_factory
        c1 = factory.getOWLClass(IRI.create("http://test/D1"))
        c2 = factory.getOWLClass(IRI.create("http://test/D2"))
        ont = adapter.owl_manager.createOntology(IRI.create("http://test/no_roles"))
        ont.addAxiom(factory.getOWLSubClassOfAxiom(c1, c2))

        dataset = ELDataset(ont, extended=True, load_normalized=False)
        self.assertIsNone(dataset.role_inclusion_dataset)
        self.assertIsNone(dataset.role_chain_dataset)
        datasets = dataset.get_gci_datasets()
        self.assertNotIn("role_inclusion", datasets)
        self.assertNotIn("role_chain", datasets)


class TestElModuleRoleAxioms(TestCase):

    def test_loss_function_dispatch(self):
        from mowl.nn import ELModule
        import torch as th

        module = ELModule()
        self.assertIn("role_inclusion", module.gci_names)
        self.assertIn("role_chain", module.gci_names)
        self.assertEqual(module.get_loss_function("role_inclusion"), module.role_inclusion_loss)
        self.assertEqual(module.get_loss_function("role_chain"), module.role_chain_loss)

        with self.assertRaises(NotImplementedError):
            module.forward(th.tensor([[0, 1]]), "role_inclusion")
        with self.assertRaises(NotImplementedError):
            module.forward(th.tensor([[0, 1, 2]]), "role_chain")

"""The annotation corpus names a class like the axiom corpus does (the IRI); the ``<IRI>`` form
(the old output of ``extract_and_save_annotation_corpus``) is the option ``entity_form="bracketed
iri"``. The corpora list their lines in a fixed order."""
import os
import tempfile
from unittest import TestCase

import mowl
mowl.init_jvm("1g")

from org.semanticweb.owlapi.model import IRI

from mowl.corpus import (extract_and_save_annotation_corpus, extract_and_save_axiom_corpus,
                         extract_annotation_corpus, extract_axiom_corpus)
from mowl.owlapi import OWLAPIAdapter

NS = "http://test/annotations/"


def annotated_ontology():
    adapter = OWLAPIAdapter()
    f = adapter.data_factory
    ont = adapter.owl_manager.createOntology(IRI.create(NS + "ont"))
    label = f.getRDFSLabel()
    for name in "CBA":
        c = f.getOWLClass(IRI.create(NS + name))
        ont.addAxiom(f.getOWLSubClassOfAxiom(c, f.getOWLClass(IRI.create(NS + "Top"))))
        ont.addAxiom(f.getOWLAnnotationAssertionAxiom(
            c.getIRI(), f.getOWLAnnotation(label, f.getOWLLiteral(f"label of {name}"))))
    a = f.getOWLNamedIndividual(IRI.create(NS + "a"))
    ont.addAxiom(f.getOWLClassAssertionAxiom(f.getOWLClass(IRI.create(NS + "Top")), a))
    ont.addAxiom(f.getOWLAnnotationAssertionAxiom(
        a.getIRI(), f.getOWLAnnotation(label, f.getOWLLiteral("an individual"))))
    return ont


def saved(ontology, **kwargs):
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        path = os.path.join(d, "annotations.txt")
        extract_and_save_annotation_corpus(ontology, path, **kwargs)
        with open(path) as f:
            return f.readlines()


class EntityForm(TestCase):

    def test_the_default_writes_the_iri_like_the_list_variant_and_the_axiom_corpus(self):
        ont = annotated_ontology()
        lines = saved(ont)
        self.assertEqual(lines, extract_annotation_corpus(ont))
        self.assertTrue(lines[0].startswith(NS + "A "), lines[0])
        self.assertFalse(any(line.startswith("<") for line in lines))
        axioms = " ".join(extract_axiom_corpus(ont))
        self.assertIn(NS + "A", axioms)          # the token the annotation sentences use

    def test_the_bracketed_form_is_the_option(self):
        ont = annotated_ontology()
        lines = saved(ont, entity_form="bracketed iri")
        self.assertEqual(lines, extract_annotation_corpus(ont, entity_form="bracketed iri"))
        classes = [line for line in lines if "label of" in line]
        self.assertTrue(all(line.startswith("<" + NS) for line in classes))
        # an individual is written as IRI in both forms
        self.assertTrue(any(line.startswith(NS + "a ") for line in lines))

    def test_an_unknown_form_is_refused(self):
        with self.assertRaises(ValueError):
            extract_annotation_corpus(annotated_ontology(), entity_form="curie")
        with self.assertRaises(ValueError):
            saved(annotated_ontology(), entity_form="curie")


class Order(TestCase):

    def test_the_corpora_are_the_same_on_every_build(self):
        builds = [annotated_ontology() for _ in range(4)]
        for corpus in (extract_annotation_corpus, extract_axiom_corpus):
            lines = [[str(s) for s in corpus(o)] for o in builds]
            self.assertTrue(all(x == lines[0] for x in lines))
            self.assertEqual(lines[0], sorted(lines[0]) if corpus is extract_axiom_corpus
                             else lines[0])
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            files = []
            for i, o in enumerate(builds):
                path = os.path.join(d, f"axioms{i}.txt")
                extract_and_save_axiom_corpus(o, path)
                with open(path) as f:
                    files.append(f.read())
            self.assertTrue(all(x == files[0] for x in files))

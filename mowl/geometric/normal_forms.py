"""EL++ normal forms and ontology normalisation for :mod:`mowl.geometric`.

Guarantees:
- Converts a mOWL :class:`~mowl.datasets.Dataset` or OWLAPI ``OWLOntology`` into
  a :class:`NormalisedOntology` containing frozen normal-form dataclasses:
  :class:`NF1` (``A <= B`` and ``A <= bot``), :class:`NF2` (``A1 and A2 <= B`` and
  ``A1 and A2 <= bot``), :class:`NF3` (``A <= exists r.B``), :class:`NF4`
  (``exists r.A <= B`` and ``exists r.A <= bot``), :class:`NF6` (``r <= s``), and
  :class:`NF7` (``r1 o r2 <= s``).
- Extracts role hierarchy (``SubObjectPropertyOf``) and role chain
  (``SubPropertyChainOf``) axioms directly from the OWLAPI ontology, splitting
  role chains of length > 2 using fresh auxiliary role IRIs
  ``http://mowl.borg/geo/aux#<sup_local>_chain_<i>``.
- Collects all class IRIs in the ontology signature, all auxiliary class IRIs
  produced by :class:`~mowl.ontology.normalize.ELNormalizer`, ``TOP``, and
  ``BOTTOM`` into ``concept_names``.
- Records the number of TBox axioms dropped during :class:`ELNormalizer`
  preprocessing in ``dropped_axioms``.

References:
- Baader, Brandt & Lutz, IJCAI 2005 (Pushing the EL Envelope — EL++ normal forms).
- Kazakov, Krötzsch & Simančík, JAR 2014 (The Incredible ELK).
- Gutiérrez-Basulto & Schockaert, KR 2018 (arXiv:1805.10461), Props. 3, 4.
- Lacerda, Ozaki & Guimarães, TGDK 2(3):2, 2024, Defs. 2, 8, 12.
- Leemhuis & Kutz, NeSy 2025 (PMLR 284:303–321), Defs. 1, 3, 6, Props. 4, 5, 8.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Union

from java.util import ArrayList
from org.semanticweb.owlapi.model import AxiomType, IRI, OWLOntology
from org.semanticweb.owlapi.model.parameters import Imports

from mowl.datasets import Dataset
from mowl.ontology.normalize import ELNormalizer
from mowl.owlapi import OWLAPIAdapter

BOTTOM = "http://www.w3.org/2002/07/owl#Nothing"
TOP = "http://www.w3.org/2002/07/owl#Thing"
AUX_NS = "http://mowl.borg/geo/aux#"


@dataclass(frozen=True)
class NF1:
    """Normal form ``sub <= sup`` (GCI0, or GCI0_BOT / NF5 when ``sup == BOTTOM``)."""

    sub: str
    sup: str


@dataclass(frozen=True)
class NF2:
    """Normal form ``sub1 and sub2 <= sup`` (GCI1, or GCI1_BOT when ``sup == BOTTOM``)."""

    sub1: str
    sub2: str
    sup: str


@dataclass(frozen=True)
class NF3:
    """Normal form ``sub <= exists role.filler`` (GCI2)."""

    sub: str
    role: str
    filler: str


@dataclass(frozen=True)
class NF4:
    """Normal form ``exists role.filler <= sup`` (GCI3, or GCI3_BOT when ``sup == BOTTOM``)."""

    role: str
    filler: str
    sup: str


@dataclass(frozen=True)
class NF6:
    """Role inclusion normal form ``sub <= sup``."""

    sub: str
    sup: str


@dataclass(frozen=True)
class NF7:
    """Role composition (chain) normal form ``r1 o r2 <= sup``."""

    r1: str
    r2: str
    sup: str


Axiom = Union[NF1, NF2, NF3, NF4, NF6, NF7]


@dataclass
class NormalisedOntology:
    """Normalised EL++ TBox and RBox produced by :func:`normalise`."""

    axioms: List[Axiom]
    concept_names: Set[str]
    role_names: Set[str]
    dropped_axioms: int


def _clean_iri(prop_obj: object) -> str:
    """Extract an unbracketed IRI string from an OWLObjectPropertyExpression."""
    if hasattr(prop_obj, "toStringID"):
        s = str(prop_obj.toStringID())
    else:
        s = str(prop_obj.toString())
    if s.startswith("<") and s.endswith(">"):
        return s[1:-1]
    return s


def _local_name(iri: str) -> str:
    """Extract the local fragment or last path segment of an IRI."""
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    if "/" in iri:
        return iri.rsplit("/", 1)[-1]
    return iri


def normalise(source: Union[Dataset, OWLOntology]) -> NormalisedOntology:
    """Normalise a mOWL :class:`~mowl.datasets.Dataset` or OWLAPI ``OWLOntology``.

    :param source: Input dataset or OWLAPI ontology.
    :return: :class:`NormalisedOntology` with NF1–NF4, NF6, NF7 axioms, concept
        and role signatures, and dropped TBox axiom count.
    """
    if isinstance(source, Dataset):
        ontology = source.ontology
    elif isinstance(source, OWLOntology):
        ontology = source
    else:
        raise TypeError(
            f"Expected mowl.datasets.Dataset or OWLOntology, got {type(source)}"
        )

    normalizer = ELNormalizer()
    orig_tbox = ontology.getTBoxAxioms(Imports.fromBoolean(True))
    preprocessed = normalizer.preprocess_ontology(ontology)
    prep_tbox = preprocessed.getTBoxAxioms(Imports.fromBoolean(True))
    dropped_axioms = max(0, int(orig_tbox.size()) - int(prep_tbox.size()))

    gci_dict = normalizer.normalize(ontology)

    axioms: List[Axiom] = []
    concept_names: Set[str] = {TOP, BOTTOM}
    role_names: Set[str] = set()

    for cls in ontology.getClassesInSignature(Imports.fromBoolean(True)):
        concept_names.add(str(cls.toStringID()))

    for prop in ontology.getObjectPropertiesInSignature(Imports.fromBoolean(True)):
        role_names.add(_clean_iri(prop))

    for gci in gci_dict["gci0"] + gci_dict["gci0_bot"]:
        sub = str(gci.subclass)
        sup = str(gci.superclass)
        axioms.append(NF1(sub, sup))
        concept_names.add(sub)
        concept_names.add(sup)

    for gci in gci_dict["gci1"] + gci_dict["gci1_bot"]:
        sub1 = str(gci.left_subclass)
        sub2 = str(gci.right_subclass)
        sup = str(gci.superclass)
        axioms.append(NF2(sub1, sub2, sup))
        concept_names.add(sub1)
        concept_names.add(sub2)
        concept_names.add(sup)

    for gci in gci_dict["gci2"]:
        sub = str(gci.subclass)
        role = str(gci.object_property)
        filler = str(gci.filler)
        axioms.append(NF3(sub, role, filler))
        concept_names.add(sub)
        concept_names.add(filler)
        role_names.add(role)

    for gci in gci_dict["gci3"] + gci_dict["gci3_bot"]:
        role = str(gci.object_property)
        filler = str(gci.filler)
        sup = str(gci.superclass)
        axioms.append(NF4(role, filler, sup))
        concept_names.add(filler)
        concept_names.add(sup)
        role_names.add(role)

    # Extract role inclusions (NF6) across imports closure
    for ont in ontology.getImportsClosure():
        for ax in ont.getAxioms(AxiomType.SUB_OBJECT_PROPERTY):
            sub_p = ax.getSubProperty()
            sup_p = ax.getSuperProperty()
            if sub_p.isAnonymous() or sup_p.isAnonymous():
                continue
            sub_r = _clean_iri(sub_p)
            sup_r = _clean_iri(sup_p)
            axioms.append(NF6(sub_r, sup_r))
            role_names.add(sub_r)
            role_names.add(sup_r)

    # Extract role chains (NF7) across imports closure, splitting chains of length > 2
    chain_counters: dict = {}
    for ont in ontology.getImportsClosure():
        for ax in ont.getAxioms(AxiomType.SUB_PROPERTY_CHAIN_OF):
            sup_p = ax.getSuperProperty()
            if sup_p.isAnonymous():
                continue
            chain_props = list(ax.getPropertyChain())
            if any(p.isAnonymous() for p in chain_props):
                continue
            chain_iris = [_clean_iri(p) for p in chain_props]
            sup_r = _clean_iri(sup_p)
            for r_iri in chain_iris:
                role_names.add(r_iri)
            role_names.add(sup_r)

            if len(chain_iris) == 0:
                continue
            elif len(chain_iris) == 1:
                axioms.append(NF6(chain_iris[0], sup_r))
            elif len(chain_iris) == 2:
                axioms.append(NF7(chain_iris[0], chain_iris[1], sup_r))
            else:
                sup_local = _local_name(sup_r)
                prev = chain_iris[0]
                for idx in range(1, len(chain_iris) - 1):
                    count = chain_counters.get(sup_local, 0) + 1
                    chain_counters[sup_local] = count
                    fresh_role = f"{AUX_NS}{sup_local}_chain_{count}"
                    while fresh_role in role_names:
                        count += 1
                        chain_counters[sup_local] = count
                        fresh_role = f"{AUX_NS}{sup_local}_chain_{count}"
                    role_names.add(fresh_role)
                    axioms.append(NF7(prev, chain_iris[idx], fresh_role))
                    prev = fresh_role
                axioms.append(NF7(prev, chain_iris[-1], sup_r))

    return NormalisedOntology(
        axioms=axioms,
        concept_names=concept_names,
        role_names=role_names,
        dropped_axioms=dropped_axioms,
    )


def to_owl(no: NormalisedOntology) -> OWLOntology:
    """Re-serialise a :class:`NormalisedOntology` into an OWLAPI ``OWLOntology``."""
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    df = adapter.data_factory
    ont = adapter.create_ontology("http://mowl.borg/geo/normalised")

    for cname in no.concept_names:
        if cname not in (TOP, BOTTOM):
            mgr.addAxiom(ont, df.getOWLDeclarationAxiom(adapter.create_class(cname)))

    for rname in no.role_names:
        mgr.addAxiom(
            ont, df.getOWLDeclarationAxiom(adapter.create_object_property(rname))
        )

    for ax in no.axioms:
        if isinstance(ax, NF1):
            c_sub = adapter.create_class(ax.sub)
            c_sup = adapter.create_class(ax.sup)
            mgr.addAxiom(ont, adapter.create_subclass_of(c_sub, c_sup))
        elif isinstance(ax, NF2):
            c_sub1 = adapter.create_class(ax.sub1)
            c_sub2 = adapter.create_class(ax.sub2)
            c_sup = adapter.create_class(ax.sup)
            mgr.addAxiom(
                ont,
                adapter.create_subclass_of(
                    adapter.create_object_intersection_of(c_sub1, c_sub2), c_sup
                ),
            )
        elif isinstance(ax, NF3):
            c_sub = adapter.create_class(ax.sub)
            prop = adapter.create_object_property(ax.role)
            c_fill = adapter.create_class(ax.filler)
            mgr.addAxiom(
                ont,
                adapter.create_subclass_of(
                    c_sub, adapter.create_object_some_values_from(prop, c_fill)
                ),
            )
        elif isinstance(ax, NF4):
            prop = adapter.create_object_property(ax.role)
            c_fill = adapter.create_class(ax.filler)
            c_sup = adapter.create_class(ax.sup)
            mgr.addAxiom(
                ont,
                adapter.create_subclass_of(
                    adapter.create_object_some_values_from(prop, c_fill), c_sup
                ),
            )
        elif isinstance(ax, NF6):
            p_sub = adapter.create_object_property(ax.sub)
            p_sup = adapter.create_object_property(ax.sup)
            mgr.addAxiom(ont, df.getOWLSubObjectPropertyOfAxiom(p_sub, p_sup))
        elif isinstance(ax, NF7):
            p1 = adapter.create_object_property(ax.r1)
            p2 = adapter.create_object_property(ax.r2)
            p_sup = adapter.create_object_property(ax.sup)
            chain = ArrayList()
            chain.add(p1)
            chain.add(p2)
            mgr.addAxiom(ont, df.getOWLSubPropertyChainOfAxiom(chain, p_sup))

    return ont

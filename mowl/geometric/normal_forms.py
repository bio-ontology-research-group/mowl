"""EL++ normal forms and ontology normalisation for :mod:`mowl.geometric`.

Guarantees:
- Converts a mOWL :class:`~mowl.datasets.Dataset` or OWLAPI ``OWLOntology`` into
  a :class:`NormalisedOntology` containing frozen normal-form dataclasses:
  :class:`NF1` (``A <= B`` and ``A <= bot``), :class:`NF2` (``A1 and A2 <= B`` and
  ``A1 and A2 <= bot``), :class:`NF3` (``A <= exists r.B``), :class:`NF4`
  (``exists r.A <= B`` and ``exists r.A <= bot``), :class:`NF6` (``r <= s``), and
  :class:`NF7` (``r1 o r2 <= s``).
- Extracts role hierarchy (``SubObjectPropertyOf``), role equivalence
  (``EquivalentObjectProperties``), role transitivity
  (``TransitiveObjectProperty``), and role chain (``SubPropertyChainOf``)
  axioms directly from the OWLAPI ontology, splitting role chains of length > 2
  using fresh auxiliary role IRIs
  ``http://mowl.borg/geo/aux#<sup_local>_chain_<i>``.
- Collects all class IRIs in the ontology signature, all auxiliary class IRIs
  produced by :class:`~mowl.ontology.normalize.ELNormalizer`, ``TOP``, and
  ``BOTTOM`` into ``concept_names``.
- Records the number of TBox and RBox axioms dropped during normalisation in
  ``dropped_axioms``.

References:
- Baader, Brandt & Lutz, IJCAI 2005 (Pushing the EL Envelope — EL++ normal forms).
- Kazakov, Krötzsch & Simančík, JAR 2014 (The Incredible ELK).
- Gutiérrez-Basulto & Schockaert, KR 2018 (arXiv:1805.10461), Props. 3, 4.
- Lacerda, Ozaki & Guimarães, TGDK 2(3):2, 2024, Defs. 2, 8, 12.
- Leemhuis & Kutz, NeSy 2025 (PMLR 284:303–321), Defs. 1, 3, 6, Props. 4, 5, 8.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import TYPE_CHECKING, List, Set, Union
import urllib.parse

from mowl.geometric.errors import UnsupportedAxiomError

if TYPE_CHECKING:  # pragma: no cover
    from org.semanticweb.owlapi.model import OWLOntology
    from mowl.datasets import Dataset

BOTTOM = "http://www.w3.org/2002/07/owl#Nothing"
TOP = "http://www.w3.org/2002/07/owl#Thing"
AUX_NS = "http://mowl.borg/geo/aux#"
NOMINAL_PREFIX = "http://mowl.borg/nominal/"


def nominal_name(individual_iri: str) -> str:
    """Encode a named individual IRI as a fresh nominal concept name.

    Constructs ``NOMINAL_PREFIX + urllib.parse.quote(str(individual_iri), safe="")``
    so that arbitrary characters in ``individual_iri`` (such as ``#``, ``/``,
    ``?``, ``%``, spaces, or non-ASCII characters) yield a valid, collision-free
    IRI string suitable for use as an atomic concept name in :class:`NF1` and
    :class:`NF3` axioms.

    Guarantees
    ----------
    - Bijective on string representations of individual IRIs:
      ``individual_of_name(nominal_name(iri)) == str(iri)`` and
      ``is_nominal_name(nominal_name(iri)) is True`` for every ``iri``.
    - Used by :func:`normalise` and :class:`~mowl.geometric.completion.CanonicalModel`
      to encode atomic ABox assertions ``C(a)`` as ``NF1(N_a, C)`` and
      ``r(a, b)`` as ``NF3(N_a, r, N_b)``, which is sound and complete for
      instance checking in :math:`\\mathcal{ELH}` (with :math:`\\bot`) without
      nominals (``ObjectOneOf``), functionality (``FunctionalObjectProperty``),
      and range (``ObjectPropertyRange``) axioms.

    Non-guarantees
    --------------
    - Does not validate that ``individual_iri`` is declared in any ontology;
      querying :meth:`~mowl.geometric.completion.CanonicalModel.instance_of` or
      :meth:`~mowl.geometric.completion.CanonicalModel.assertion_contradicted`
      with an unknown individual or class IRI returns ``False``.
    - The nominal encoding is sound and complete for instance checking only for
      :math:`\\mathcal{ELH}` without nominals, functionality, and range axioms.
      Unsupported ABox assertions (complex ``ClassAssertion``,
      ``SameIndividual``, ``DifferentIndividuals``, ``ObjectPropertyAssertion``
      with an inverse property, assertions on anonymous individuals, and
      negative or data assertions) raise
      :class:`~mowl.geometric.errors.UnsupportedAxiomError` during
      :func:`normalise`.

    Parameters
    ----------
    individual_iri : str
        Unbracketed IRI string of a named individual.

    Returns
    -------
    str
        Fresh nominal concept IRI ``NOMINAL_PREFIX + <percent-encoded IRI>``.
    """
    return NOMINAL_PREFIX + urllib.parse.quote(str(individual_iri), safe="")


def is_nominal_name(name: str) -> bool:
    """Test whether a concept name IRI is a nominal name produced by :func:`nominal_name`.

    Guarantees
    ----------
    - Returns ``True`` if and only if ``str(name).startswith(NOMINAL_PREFIX)``.
    - Partitions canonical-model concept identifiers into nominal names
      (individuals encoded via :func:`nominal_name`, included in
      :attr:`~mowl.geometric.completion.CanonicalModel.names` so region
      builders treat them as names) and non-nominal concept names
      (:attr:`~mowl.geometric.completion.CanonicalModel.concept_names`).
    - Sound and complete instance checking via nominal names holds for
      :math:`\\mathcal{ELH}` without nominals, functionality, and range axioms.

    Non-guarantees
    --------------
    - Checks only the ``NOMINAL_PREFIX`` prefix; does not verify whether the
      decoded individual IRI occurs in a specific
      :class:`~mowl.geometric.completion.CanonicalModel` (unknown IRIs return
      ``False`` in ``instance_of`` and ``assertion_contradicted``).
    - Unsupported ABox assertions (complex ``ClassAssertion``,
      ``SameIndividual``, ``DifferentIndividuals``, ``ObjectPropertyAssertion``
      with an inverse property, assertions on anonymous individuals, and
      negative or data assertions) are rejected by :func:`normalise` with
      :class:`~mowl.geometric.errors.UnsupportedAxiomError`.

    Parameters
    ----------
    name : str
        Candidate concept or nominal IRI string.

    Returns
    -------
    bool
        ``True`` iff ``name`` starts with :data:`NOMINAL_PREFIX`.
    """
    return str(name).startswith(NOMINAL_PREFIX)


def individual_of_name(name: str) -> str:
    """Decode the original individual IRI from a nominal concept name.

    Inverts :func:`nominal_name` by stripping :data:`NOMINAL_PREFIX` and
    percent-decoding the remainder via :func:`urllib.parse.unquote`.

    Guarantees
    ----------
    - Exact left-inverse of :func:`nominal_name`:
      ``individual_of_name(nominal_name(a)) == str(a)`` for any string ``a``.
    - Raises :class:`ValueError` whenever ``not is_nominal_name(name)``.
    - Supports the nominal ABox encoding of :func:`normalise`, which is sound
      and complete for instance checking in :math:`\\mathcal{ELH}` without
      nominals, functionality, and range axioms.

    Non-guarantees
    --------------
    - Does not check whether the decoded individual IRI belongs to a given
      ontology (unknown IRIs evaluate to ``False`` in
      :meth:`~mowl.geometric.completion.CanonicalModel.instance_of` and
      :meth:`~mowl.geometric.completion.CanonicalModel.assertion_contradicted`).
    - Sound and complete instance checking via nominal names holds only for
      :math:`\\mathcal{ELH}` without nominals, functionality, and range axioms;
      complex ``ClassAssertion``, ``SameIndividual``, ``DifferentIndividuals``,
      ``ObjectPropertyAssertion`` with an inverse property, assertions on
      anonymous individuals, and negative/data assertions raise
      :class:`~mowl.geometric.errors.UnsupportedAxiomError` in :func:`normalise`.

    Parameters
    ----------
    name : str
        Nominal concept name starting with :data:`NOMINAL_PREFIX`.

    Returns
    -------
    str
        Decoded individual IRI string.

    Raises
    ------
    ValueError
        If ``name`` does not start with :data:`NOMINAL_PREFIX`.
    """
    if not is_nominal_name(name):
        raise ValueError(
            f"Expected a nominal name starting with {NOMINAL_PREFIX!r}, got {name!r}"
        )
    return urllib.parse.unquote(str(name)[len(NOMINAL_PREFIX):])


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

    Guarantees
    ----------
    - *TBox (via* :class:`~mowl.ontology.normalize.ELNormalizer` *):*
      ``SubClassOf``, ``EquivalentClasses``, ``DisjointClasses``, and
      ``ObjectPropertyDomain(r, C)`` (translated into ``exists r.owl:Thing <= C``,
      producing :class:`NF4` ``(r, TOP, A)`` when ``C`` is a concept name ``A``)
      constructed over ``EL_bot`` concepts (named classes, ``owl:Thing``,
      ``owl:Nothing``, ``ObjectIntersectionOf``, and ``ObjectSomeValuesFrom``
      over named object properties) are normalised into :class:`NF1`,
      :class:`NF2`, :class:`NF3`, and :class:`NF4`.
    - *RBox (extracted across the imports closure):*
      - ``SubObjectPropertyOf(r, s)`` with named ``r, s`` -> :class:`NF6` ``(r, s)``.
      - ``EquivalentObjectProperties(r_1, ..., r_k)`` with named ``r_i`` ->
        pairwise :class:`NF6` ``(r_i, r_j)`` and :class:`NF6` ``(r_j, r_i)`` for
        all ``i != j``.
      - ``TransitiveObjectProperty(r)`` with named ``r`` -> :class:`NF7` ``(r, r, r)``.
      - ``SubPropertyChainOf([r], s)`` with named ``r, s`` -> :class:`NF6` ``(r, s)``.
      - ``SubPropertyChainOf([r_1, r_2], s)`` with named ``r_1, r_2, s`` ->
        :class:`NF7` ``(r_1, r_2, s)``.
      - ``SubPropertyChainOf([r_1, ..., r_k], s)`` (``k > 2``) with named roles ->
        split into ``k - 1`` :class:`NF7` axioms using fresh auxiliary roles
        ``http://mowl.borg/geo/aux#<s_local>_chain_<i>``.
    - *ABox (extracted across the imports closure, individuals as fresh nominal names):*
      - Each named individual ``a`` is mapped to ``N_a = nominal_name(a)`` and
        included in ``concept_names``.
      - Atomic ``ClassAssertion(C, a)`` with a named class ``C`` and named
        individual ``a`` -> :class:`NF1` ``(N_a, C)``.
      - ``ObjectPropertyAssertion(r, a, b)`` with a named object property ``r``
        and named individuals ``a, b`` -> :class:`NF3` ``(N_a, r, N_b)``.
      - This nominal encoding is sound and complete for instance checking in
        :math:`\\mathcal{ELH}` (with :math:`\\bot`) without nominals
        (``ObjectOneOf``), functionality (``FunctionalObjectProperty``), and
        range (``ObjectPropertyRange``) axioms.

    Non-guarantees
    --------------
    - Sound and complete for instance checking only for :math:`\\mathcal{ELH}`
      without nominals, functionality, and range axioms.
    - Querying :meth:`~mowl.geometric.completion.CanonicalModel.instance_of` or
      :meth:`~mowl.geometric.completion.CanonicalModel.assertion_contradicted`
      with unknown individual or class IRIs returns ``False``.
    - *ABox assertions that raise*
      :class:`~mowl.geometric.errors.UnsupportedAxiomError` *(with the axiom text;
      no silent ABox drops):*
      - ``ClassAssertion`` with a complex (anonymous) class expression.
      - ``ClassAssertion`` on an anonymous individual.
      - ``ObjectPropertyAssertion`` with an inverse property
        (``ObjectInverseOf``).
      - ``ObjectPropertyAssertion`` on an anonymous subject or object individual.
      - ``SameIndividual`` and ``DifferentIndividuals``.
      - Negative and data property assertions
        (``NegativeObjectPropertyAssertion``, ``DataPropertyAssertion``,
        ``NegativeDataPropertyAssertion``).
    - *Dropped TBox and RBox axioms (each increments ``dropped_axioms`` by 1):*
      - Any TBox axiom filtered out by :meth:`ELNormalizer.preprocess_ontology`
        (axioms containing ``ObjectUnionOf``, ``ObjectComplementOf``,
        ``ObjectAllValuesFrom``, ``ObjectMinCardinality``, ``ObjectMaxCardinality``,
        ``ObjectExactCardinality``, ``ObjectHasSelf``, ``ObjectOneOf``, SWRL rules,
        etc.) or containing ``ObjectInverseOf``.
      - ``ObjectPropertyRange`` and any other OWLAPI TBox axiom type not
        converted by ``ELNormalizer`` (``FunctionalObjectProperty``,
        ``InverseFunctionalObjectProperty``, ``DisjointUnion``, ``HasKey``,
        ``DatatypeDefinition``, ``DataPropertyDomain``, ``DataPropertyRange``,
        ``FunctionalDataProperty``).
      - ``SubObjectPropertyOf``, ``EquivalentObjectProperties``,
        ``TransitiveObjectProperty``, or ``SubPropertyChainOf`` axioms that
        contain any inverse/anonymous object property expression, or
        ``EquivalentObjectProperties`` with fewer than 2 distinct properties, or
        ``SubPropertyChainOf`` with an empty property chain, and any other
        OWLAPI RBox axiom type (``InverseObjectProperties``,
        ``SymmetricObjectProperty``, ``AsymmetricObjectProperty``,
        ``ReflexiveObjectProperty``, ``IrreflexiveObjectProperty``,
        ``DisjointObjectProperties``, ``SubDataPropertyOf``,
        ``EquivalentDataProperties``, ``DisjointDataProperties``).

    Parameters
    ----------
    source : Union[Dataset, OWLOntology]
        Input mOWL dataset or OWLAPI ontology.

    Returns
    -------
    NormalisedOntology
        Normalised ontology with ``NF1``–``NF4``, ``NF6``, ``NF7`` axioms,
        concept and role signatures (including nominal concept names), and total
        dropped TBox + RBox axiom count.

    Raises
    ------
    TypeError
        If ``source`` is neither a :class:`~mowl.datasets.Dataset` nor an
        ``OWLOntology``.
    UnsupportedAxiomError
        If the ontology contains an unsupported ABox axiom (complex
        ``ClassAssertion``, ``SameIndividual``, ``DifferentIndividuals``,
        ``ObjectPropertyAssertion`` with an inverse property, an assertion on an
        anonymous individual, or a negative/data assertion).
    """
    from java.util import HashSet
    from org.semanticweb.owlapi.model import AxiomType, OWLOntology
    from org.semanticweb.owlapi.model.parameters import Imports

    from mowl.datasets import Dataset
    from mowl.ontology.normalize import ELNormalizer
    from mowl.owlapi import OWLAPIAdapter

    supported_tbox_types = {
        AxiomType.SUBCLASS_OF,
        AxiomType.EQUIVALENT_CLASSES,
        AxiomType.DISJOINT_CLASSES,
        AxiomType.OBJECT_PROPERTY_DOMAIN,
    }

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

    filtered_tbox = HashSet()
    for ax in prep_tbox:
        ax_type = ax.getAxiomType()
        if ax_type not in supported_tbox_types or "ObjectInverseOf" in str(ax.toString()):
            dropped_axioms += 1
        else:
            filtered_tbox.add(ax)

    clean_tbox_ont = OWLAPIAdapter().owl_manager.createOntology(filtered_tbox)
    gci_dict = normalizer.normalize(clean_tbox_ont, use_cache=False)

    axioms: List[Axiom] = []
    concept_names: Set[str] = {TOP, BOTTOM}
    role_names: Set[str] = set()

    for cls in ontology.getClassesInSignature(Imports.fromBoolean(True)):
        concept_names.add(str(cls.toStringID()))

    for prop in ontology.getObjectPropertiesInSignature(Imports.fromBoolean(True)):
        role_names.add(_clean_iri(prop))

    for ind in ontology.getIndividualsInSignature(Imports.fromBoolean(True)):
        concept_names.add(nominal_name(_clean_iri(ind)))

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

    # Extract RBox axioms across the imports closure and count unsupported RBox axioms
    rbox_axioms = ontology.getRBoxAxioms(Imports.fromBoolean(True))
    chain_counters: dict = {}
    for ax in rbox_axioms:
        ax_type = ax.getAxiomType()
        if ax_type == AxiomType.SUB_OBJECT_PROPERTY:
            sub_p = ax.getSubProperty()
            sup_p = ax.getSuperProperty()
            if sub_p.isAnonymous() or sup_p.isAnonymous():
                dropped_axioms += 1
                continue
            sub_r = _clean_iri(sub_p)
            sup_r = _clean_iri(sup_p)
            axioms.append(NF6(sub_r, sup_r))
            role_names.add(sub_r)
            role_names.add(sup_r)
        elif ax_type == AxiomType.EQUIVALENT_OBJECT_PROPERTIES:
            props = list(ax.getProperties())
            if len(props) < 2 or any(p.isAnonymous() for p in props):
                dropped_axioms += 1
                continue
            prop_iris = sorted({_clean_iri(p) for p in props})
            if len(prop_iris) < 2:
                dropped_axioms += 1
                continue
            for r_iri in prop_iris:
                role_names.add(r_iri)
            for r1, r2 in itertools.permutations(prop_iris, 2):
                axioms.append(NF6(r1, r2))
        elif ax_type == AxiomType.TRANSITIVE_OBJECT_PROPERTY:
            prop = ax.getProperty()
            if prop.isAnonymous():
                dropped_axioms += 1
                continue
            r_iri = _clean_iri(prop)
            axioms.append(NF7(r_iri, r_iri, r_iri))
            role_names.add(r_iri)
        elif ax_type == AxiomType.SUB_PROPERTY_CHAIN_OF:
            sup_p = ax.getSuperProperty()
            chain_props = list(ax.getPropertyChain())
            if (
                sup_p.isAnonymous()
                or len(chain_props) == 0
                or any(p.isAnonymous() for p in chain_props)
            ):
                dropped_axioms += 1
                continue
            chain_iris = [_clean_iri(p) for p in chain_props]
            sup_r = _clean_iri(sup_p)
            for r_iri in chain_iris:
                role_names.add(r_iri)
            role_names.add(sup_r)

            if len(chain_iris) == 1:
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
        else:
            dropped_axioms += 1

    # Extract ABox axioms across the imports closure (individuals as fresh nominal names)
    abox_axioms = ontology.getABoxAxioms(Imports.fromBoolean(True))
    for ax in abox_axioms:
        ax_type = ax.getAxiomType()
        if ax_type == AxiomType.CLASS_ASSERTION:
            ce = ax.getClassExpression()
            ind = ax.getIndividual()
            if ind.isAnonymous():
                raise UnsupportedAxiomError(
                    f"Unsupported ClassAssertion on anonymous individual: {ax}"
                )
            if ce.isAnonymous():
                raise UnsupportedAxiomError(
                    f"Unsupported complex ClassAssertion: {ax}"
                )
            nom_a = nominal_name(_clean_iri(ind))
            cls_iri = _clean_iri(ce)
            axioms.append(NF1(nom_a, cls_iri))
            concept_names.add(nom_a)
            concept_names.add(cls_iri)
        elif ax_type == AxiomType.OBJECT_PROPERTY_ASSERTION:
            prop = ax.getProperty()
            subj = ax.getSubject()
            obj = ax.getObject()
            if prop.isAnonymous():
                raise UnsupportedAxiomError(
                    f"Unsupported ObjectPropertyAssertion with inverse property: {ax}"
                )
            if subj.isAnonymous() or obj.isAnonymous():
                raise UnsupportedAxiomError(
                    f"Unsupported ObjectPropertyAssertion on anonymous individual: {ax}"
                )
            r_iri = _clean_iri(prop)
            nom_a = nominal_name(_clean_iri(subj))
            nom_b = nominal_name(_clean_iri(obj))
            axioms.append(NF3(nom_a, r_iri, nom_b))
            concept_names.add(nom_a)
            concept_names.add(nom_b)
            role_names.add(r_iri)
        else:
            raise UnsupportedAxiomError(
                f"Unsupported ABox axiom ({ax_type}): {ax}"
            )

    return NormalisedOntology(
        axioms=axioms,
        concept_names=concept_names,
        role_names=role_names,
        dropped_axioms=dropped_axioms,
    )


def to_owl(no: NormalisedOntology) -> OWLOntology:
    """Re-serialise a :class:`NormalisedOntology` into an OWLAPI ``OWLOntology``."""
    from java.util import ArrayList

    from mowl.owlapi import OWLAPIAdapter

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

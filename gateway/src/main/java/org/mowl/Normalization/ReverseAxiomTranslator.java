package org.mowl.Normalization;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

import org.semanticweb.owlapi.model.IRI;
import org.semanticweb.owlapi.model.OWLAnnotation;
import org.semanticweb.owlapi.model.OWLAnnotationProperty;
import org.semanticweb.owlapi.model.OWLAnnotationValue;
import org.semanticweb.owlapi.model.OWLAxiom;
import org.semanticweb.owlapi.model.OWLClass;
import org.semanticweb.owlapi.model.OWLClassExpression;
import org.semanticweb.owlapi.model.OWLDataFactory;
import org.semanticweb.owlapi.model.OWLNamedIndividual;
import org.semanticweb.owlapi.model.OWLObjectOneOf;
import org.semanticweb.owlapi.model.OWLObjectProperty;

import de.tudresden.inf.lat.jcel.coreontology.axiom.FunctObjectPropAxiom;
import de.tudresden.inf.lat.jcel.coreontology.axiom.GCI0Axiom;
import de.tudresden.inf.lat.jcel.coreontology.axiom.GCI1Axiom;
import de.tudresden.inf.lat.jcel.coreontology.axiom.GCI2Axiom;
import de.tudresden.inf.lat.jcel.coreontology.axiom.GCI3Axiom;
import de.tudresden.inf.lat.jcel.coreontology.axiom.IntegerAnnotation;
import de.tudresden.inf.lat.jcel.coreontology.axiom.NominalAxiom;
import de.tudresden.inf.lat.jcel.coreontology.axiom.NormalizedIntegerAxiomVisitor;
import de.tudresden.inf.lat.jcel.coreontology.axiom.RI1Axiom;
import de.tudresden.inf.lat.jcel.coreontology.axiom.RI2Axiom;
import de.tudresden.inf.lat.jcel.coreontology.axiom.RI3Axiom;
import de.tudresden.inf.lat.jcel.coreontology.axiom.RangeAxiom;
import de.tudresden.inf.lat.jcel.coreontology.datatype.IntegerEntityManager;
import de.tudresden.inf.lat.jcel.owlapi.translator.TranslationRepository;

/**
 * Translates normalized integer axioms produced by jcel back into OWL axioms.
 *
 * <p>
 * This plays the same role as jcel's own {@code ReverseAxiomTranslator}, but it
 * knows about the auxiliary entities that the normalization rules introduce.
 * jcel's {@code TranslationRepository} only maps identifiers that came from the
 * source ontology; asking it for an identifier minted during normalization
 * raises a {@code TranslationException}, which makes the whole axiom
 * untranslatable and, in practice, silently dropped.
 *
 * <p>
 * Here, an identifier that the repository does not know is taken to be an
 * auxiliary entity and a fresh OWL entity is minted for it in the
 * {@link #AUX_NAMESPACE} namespace. Identifiers are minted once and cached, so
 * every occurrence of the same auxiliary concept across the normalized axioms
 * maps to the same OWL entity.
 *
 * @see ELNormalizer
 */
public class ReverseAxiomTranslator implements NormalizedIntegerAxiomVisitor<OWLAxiom> {

    /** Namespace of every entity minted for an auxiliary jcel identifier. */
    public static final String AUX_NAMESPACE = "http://mowl.borg/el_normalization#";

    /** Prefix of auxiliary concept names, e.g. {@code ..#aux_11}. */
    public static final String AUX_CLASS_PREFIX = AUX_NAMESPACE + "aux_";

    /** Prefix of auxiliary object property names, e.g. {@code ..#aux_prop_10}. */
    public static final String AUX_OBJECT_PROPERTY_PREFIX = AUX_NAMESPACE + "aux_prop_";

    /** Prefix of auxiliary individual names, e.g. {@code ..#aux_ind_12}. */
    public static final String AUX_INDIVIDUAL_PREFIX = AUX_NAMESPACE + "aux_ind_";

    private final TranslationRepository repository;
    private final IntegerEntityManager entityManager;
    private final OWLDataFactory dataFactory;

    private final Map<Integer, OWLClass> auxClasses = new HashMap<>();
    private final Map<Integer, OWLObjectProperty> auxObjectProperties = new HashMap<>();
    private final Map<Integer, OWLNamedIndividual> auxIndividuals = new HashMap<>();

    /**
     * Creates a new reverse axiom translator.
     *
     * @param repository
     *            translation repository holding the identifiers of the entities
     *            that occur in the source ontology
     * @param entityManager
     *            entity manager shared by the translator and the normalizer
     * @param dataFactory
     *            factory used to build the resulting OWL axioms
     */
    public ReverseAxiomTranslator(TranslationRepository repository, IntegerEntityManager entityManager,
            OWLDataFactory dataFactory) {
        Objects.requireNonNull(repository);
        Objects.requireNonNull(entityManager);
        Objects.requireNonNull(dataFactory);
        this.repository = repository;
        this.entityManager = entityManager;
        this.dataFactory = dataFactory;
    }

    /**
     * Returns the auxiliary concepts minted so far, keyed by jcel identifier.
     *
     * @return the auxiliary concepts introduced during normalization
     */
    public Map<Integer, OWLClass> getAuxiliaryClasses() {
        return new HashMap<>(this.auxClasses);
    }

    /**
     * Returns the auxiliary object properties minted so far, keyed by jcel
     * identifier. These are the inverse properties that jcel introduces for
     * every object property in the signature.
     *
     * @return the auxiliary object properties introduced during normalization
     */
    public Map<Integer, OWLObjectProperty> getAuxiliaryObjectProperties() {
        return new HashMap<>(this.auxObjectProperties);
    }

    private OWLClass getOWLClass(Integer id) {
        Optional<OWLClass> ret = this.repository.getOptOWLClass(id);
        if (ret.isPresent()) {
            return ret.get();
        }
        return this.auxClasses.computeIfAbsent(id,
                key -> this.dataFactory.getOWLClass(IRI.create(AUX_CLASS_PREFIX + key)));
    }

    private OWLObjectProperty getOWLObjectProperty(Integer id) {
        Optional<OWLObjectProperty> ret = this.repository.getOptOWLObjectProperty(id);
        if (ret.isPresent()) {
            return ret.get();
        }
        return this.auxObjectProperties.computeIfAbsent(id,
                key -> this.dataFactory.getOWLObjectProperty(IRI.create(AUX_OBJECT_PROPERTY_PREFIX + key)));
    }

    private OWLNamedIndividual getOWLNamedIndividual(Integer id) {
        Optional<OWLNamedIndividual> ret = this.repository.getOptOWLNamedIndividual(id);
        if (ret.isPresent()) {
            return ret.get();
        }
        return this.auxIndividuals.computeIfAbsent(id,
                key -> this.dataFactory.getOWLNamedIndividual(IRI.create(AUX_INDIVIDUAL_PREFIX + key)));
    }

    @Override
    public OWLAxiom visit(FunctObjectPropAxiom axiom) {
        Objects.requireNonNull(axiom);
        OWLObjectProperty owlProperty = getOWLObjectProperty(axiom.getProperty());
        return this.dataFactory.getOWLFunctionalObjectPropertyAxiom(owlProperty,
                translateAnnotations(axiom.getAnnotations()));
    }

    @Override
    public OWLAxiom visit(GCI0Axiom axiom) {
        Objects.requireNonNull(axiom);
        OWLClass owlSubClass = getOWLClass(axiom.getSubClass());
        OWLClass owlSuperClass = getOWLClass(axiom.getSuperClass());
        return this.dataFactory.getOWLSubClassOfAxiom(owlSubClass, owlSuperClass,
                translateAnnotations(axiom.getAnnotations()));
    }

    @Override
    public OWLAxiom visit(GCI1Axiom axiom) {
        Objects.requireNonNull(axiom);
        OWLClass owlLeftSubClass = getOWLClass(axiom.getLeftSubClass());
        OWLClass owlRightSubClass = getOWLClass(axiom.getRightSubClass());
        OWLClass owlSuperClass = getOWLClass(axiom.getSuperClass());
        Set<OWLClass> operands = new HashSet<>();
        operands.add(owlLeftSubClass);
        operands.add(owlRightSubClass);
        OWLClassExpression owlObjectIntersectionOf = this.dataFactory.getOWLObjectIntersectionOf(operands);
        return this.dataFactory.getOWLSubClassOfAxiom(owlObjectIntersectionOf, owlSuperClass,
                translateAnnotations(axiom.getAnnotations()));
    }

    @Override
    public OWLAxiom visit(GCI2Axiom axiom) {
        Objects.requireNonNull(axiom);
        OWLClass owlSubClass = getOWLClass(axiom.getSubClass());
        OWLClass owlClassInSuperClass = getOWLClass(axiom.getClassInSuperClass());
        OWLObjectProperty owlObjectProperty = getOWLObjectProperty(axiom.getPropertyInSuperClass());
        OWLClassExpression owlObjectSomeValuesFrom = this.dataFactory.getOWLObjectSomeValuesFrom(owlObjectProperty,
                owlClassInSuperClass);
        return this.dataFactory.getOWLSubClassOfAxiom(owlSubClass, owlObjectSomeValuesFrom,
                translateAnnotations(axiom.getAnnotations()));
    }

    @Override
    public OWLAxiom visit(GCI3Axiom axiom) {
        Objects.requireNonNull(axiom);
        OWLClass owlSuperClass = getOWLClass(axiom.getSuperClass());
        OWLClass owlClassInSubClass = getOWLClass(axiom.getClassInSubClass());
        OWLObjectProperty owlObjectProperty = getOWLObjectProperty(axiom.getPropertyInSubClass());
        OWLClassExpression owlObjectSomeValuesFrom = this.dataFactory.getOWLObjectSomeValuesFrom(owlObjectProperty,
                owlClassInSubClass);
        return this.dataFactory.getOWLSubClassOfAxiom(owlObjectSomeValuesFrom, owlSuperClass,
                translateAnnotations(axiom.getAnnotations()));
    }

    @Override
    public OWLAxiom visit(NominalAxiom axiom) {
        Objects.requireNonNull(axiom);
        OWLNamedIndividual owlIndividual = getOWLNamedIndividual(axiom.getIndividual());
        OWLClass owlClass = getOWLClass(axiom.getClassExpression());
        OWLObjectOneOf owlObjectOneOf = this.dataFactory.getOWLObjectOneOf(owlIndividual);
        Set<OWLClassExpression> owlClassExpressions = new HashSet<>();
        owlClassExpressions.add(owlObjectOneOf);
        owlClassExpressions.add(owlClass);
        return this.dataFactory.getOWLEquivalentClassesAxiom(owlClassExpressions,
                translateAnnotations(axiom.getAnnotations()));
    }

    @Override
    public OWLAxiom visit(RangeAxiom axiom) {
        Objects.requireNonNull(axiom);
        OWLObjectProperty owlObjectProperty = getOWLObjectProperty(axiom.getProperty());
        OWLClass owlClass = getOWLClass(axiom.getRange());
        return this.dataFactory.getOWLObjectPropertyRangeAxiom(owlObjectProperty, owlClass,
                translateAnnotations(axiom.getAnnotations()));
    }

    @Override
    public OWLAxiom visit(RI1Axiom axiom) {
        Objects.requireNonNull(axiom);
        OWLObjectProperty owlSuperProperty = getOWLObjectProperty(axiom.getSuperProperty());
        List<OWLObjectProperty> owlPropertyList = new ArrayList<>();
        return this.dataFactory.getOWLSubPropertyChainOfAxiom(owlPropertyList, owlSuperProperty,
                translateAnnotations(axiom.getAnnotations()));
    }

    @Override
    public OWLAxiom visit(RI2Axiom axiom) {
        Objects.requireNonNull(axiom);
        OWLObjectProperty owlSubProperty = getOWLObjectProperty(axiom.getSubProperty());
        OWLObjectProperty owlSuperProperty = getOWLObjectProperty(axiom.getSuperProperty());
        return this.dataFactory.getOWLSubObjectPropertyOfAxiom(owlSubProperty, owlSuperProperty,
                translateAnnotations(axiom.getAnnotations()));
    }

    @Override
    public OWLAxiom visit(RI3Axiom axiom) {
        Objects.requireNonNull(axiom);
        OWLObjectProperty owlLeftSubProperty = getOWLObjectProperty(axiom.getLeftSubProperty());
        OWLObjectProperty owlRightSubProperty = getOWLObjectProperty(axiom.getRightSubProperty());
        OWLObjectProperty owlSuperProperty = getOWLObjectProperty(axiom.getSuperProperty());
        List<OWLObjectProperty> owlPropertyList = new ArrayList<>();
        owlPropertyList.add(owlLeftSubProperty);
        owlPropertyList.add(owlRightSubProperty);
        return this.dataFactory.getOWLSubPropertyChainOfAxiom(owlPropertyList, owlSuperProperty,
                translateAnnotations(axiom.getAnnotations()));
    }

    /**
     * Translates the annotations of a normalized axiom, dropping any annotation
     * whose property or value is not in the repository. Auxiliary entities are
     * never annotated, so there is nothing sensible to mint for them.
     */
    private Set<OWLAnnotation> translateAnnotations(Set<IntegerAnnotation> annotations) {
        Objects.requireNonNull(annotations);
        Set<OWLAnnotation> owlAnnotations = new HashSet<>();
        for (IntegerAnnotation annotation : annotations) {
            Optional<OWLAnnotationProperty> property = this.repository
                    .getOptOWLAnnotationProperty(annotation.getAnnotationProperty());
            Optional<OWLAnnotationValue> value = this.repository
                    .getOptOWLAnnotationValue(annotation.getAnnotationValue());
            if (property.isPresent() && value.isPresent()) {
                owlAnnotations.add(this.dataFactory.getOWLAnnotation(property.get(), value.get()));
            }
        }
        return owlAnnotations;
    }

}

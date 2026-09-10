package org.mowl.Normalization;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

import org.semanticweb.owlapi.model.OWLAxiom;
import org.semanticweb.owlapi.model.OWLDataFactory;
import org.semanticweb.owlapi.model.OWLOntology;

import de.tudresden.inf.lat.jcel.coreontology.axiom.NormalizedIntegerAxiom;
import de.tudresden.inf.lat.jcel.ontology.axiom.complex.ComplexIntegerAxiom;
import de.tudresden.inf.lat.jcel.ontology.axiom.extension.IntegerOntologyObjectFactory;
import de.tudresden.inf.lat.jcel.ontology.axiom.extension.IntegerOntologyObjectFactoryImpl;
import de.tudresden.inf.lat.jcel.ontology.normalization.OntologyNormalizer;
import de.tudresden.inf.lat.jcel.owlapi.translator.TranslationRepository;
import de.tudresden.inf.lat.jcel.owlapi.translator.Translator;

/**
 * Normalizes an OWL ontology into the {@code EL} normal forms using jcel,
 * without the identifier collision that jcel 0.24.1 exhibits when it is driven
 * the way mOWL drives it.
 *
 * <p>
 * jcel maps every entity to an integer. A {@code Translator} assigns
 * identifiers 6, 7, 8, ... to the entities of the source ontology, and the
 * normalization rules mint further identifiers for the auxiliary concepts they
 * introduce (NR-2.2, NR-2.3, NR-3.1, NR-3.2, ...). Those two steps must draw
 * from the same pool. If each is given its own
 * {@code IntegerOntologyObjectFactory}, each gets its own
 * {@code IntegerEntityManager}, both counters start at
 * {@code firstUsableIdentifier == 6}, and the auxiliary concepts collide with
 * real classes. For {@code C ⊑ ∃r.(D ⊓ E)}, jcel 0.24.1 then yields
 *
 * <pre>
 * C ⊑ ∃r.D,  D ⊑ D,  D ⊑ E
 * </pre>
 *
 * instead of
 *
 * <pre>
 * C ⊑ ∃r.A,  A ⊑ D,  A ⊑ E
 * </pre>
 *
 * (asserting {@code D ⊑ E} between two classes of the source ontology, which
 * the input does not entail).
 *
 * <p>
 * The fix has two halves, and both are needed:
 *
 * <ol>
 * <li>The {@code Translator} and the {@code OntologyNormalizer} are given the
 * <em>same</em> {@code IntegerOntologyObjectFactory}, so auxiliary concepts get
 * identifiers above every identifier already handed out and cannot alias a real
 * class. (jcel pull request #12 achieves the same by adding a {@code startId}
 * to {@code IntegerEntityManagerImpl}; sharing the factory needs no fork.)</li>
 * <li>Reverse translation resolves those auxiliary identifiers instead of
 * failing on them: see {@link ReverseAxiomTranslator}. Without this, the fix
 * above would merely turn silently wrong axioms into silently missing
 * ones.</li>
 * </ol>
 *
 * <p>
 * The ontology is expected to have been preprocessed already, i.e. to contain
 * only axioms that jcel can translate.
 *
 * @see ReverseAxiomTranslator
 */
public class ELNormalizer {

    /**
     * Normalizes an ontology into the {@code EL} normal forms.
     *
     * @param ontology
     *            ontology to normalize, containing only axioms translatable by
     *            jcel
     * @return the normalized axioms
     */
    public Set<OWLAxiom> normalize(OWLOntology ontology) {
        Objects.requireNonNull(ontology);
        OWLDataFactory dataFactory = ontology.getOWLOntologyManager().getOWLDataFactory();

        IntegerOntologyObjectFactory factory = new IntegerOntologyObjectFactoryImpl();
        Translator translator = new Translator(dataFactory, factory);
        TranslationRepository repository = translator.getTranslationRepository();

        Set<OWLAxiom> owlAxioms = new HashSet<>();
        owlAxioms.addAll(ontology.getAxioms());
        repository.addAxiomEntities(ontology);
        for (OWLOntology importedOntology : ontology.getImportsClosure()) {
            owlAxioms.addAll(importedOntology.getAxioms());
            repository.addAxiomEntities(importedOntology);
        }

        Set<ComplexIntegerAxiom> integerAxioms = translator.translateSA(owlAxioms);

        // Reusing `factory` here is the fix: the normalizer mints auxiliary
        // identifiers from the counter the translator has already advanced.
        Set<NormalizedIntegerAxiom> normalizedAxioms = new OntologyNormalizer().normalize(integerAxioms, factory);

        ReverseAxiomTranslator reverseTranslator = new ReverseAxiomTranslator(repository, factory.getEntityManager(),
                dataFactory);

        Set<OWLAxiom> ret = new HashSet<>();
        for (NormalizedIntegerAxiom normalizedAxiom : normalizedAxioms) {
            ret.add(normalizedAxiom.accept(reverseTranslator));
        }
        return ret;
    }

}

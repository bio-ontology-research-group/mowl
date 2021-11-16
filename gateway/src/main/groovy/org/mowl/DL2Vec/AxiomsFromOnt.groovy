package org.mowl.DL2Vec

import org.semanticweb.owlapi.model.parameters.*;
import org.semanticweb.elk.owlapi.ElkReasonerFactory;
import org.semanticweb.elk.owlapi.ElkReasonerConfiguration;
import org.semanticweb.elk.reasoner.config.*;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.reasoner.*;
import org.semanticweb.owlapi.reasoner.structural.StructuralReasoner
import org.semanticweb.owlapi.vocab.OWLRDFVocabulary;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.io.*;
import org.semanticweb.owlapi.owllink.*;
import org.semanticweb.owlapi.util.*;
import org.semanticweb.owlapi.search.*;
import org.semanticweb.owlapi.manchestersyntax.renderer.*;
import org.semanticweb.owlapi.reasoner.structural.*;
import uk.ac.manchester.cs.owlapi.modularity.ModuleType;
import uk.ac.manchester.cs.owlapi.modularity.SyntacticLocalityModuleExtractor;
import org.semanticweb.owlapi.manchestersyntax.renderer.*;
import java.io.*;
import java.io.PrintWriter;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.InferenceType;
import org.semanticweb.owlapi.util.InferredAxiomGenerator;
import org.semanticweb.owlapi.util.InferredOntologyGenerator;
import org.semanticweb.owlapi.util.InferredSubClassAxiomGenerator;
import org.semanticweb.owlapi.util.InferredEquivalentClassAxiomGenerator;
import org.semanticweb.owlapi.reasoner.OWLReasonerFactory;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.owlapi.reasoner.OWLReasonerConfiguration;


class AxiomsFromOnt {

    OWLOntology ontology
    String chosenReasoner
    ArrayList<String> axioms_orig
    ArrayList<String> axioms_inf
    ArrayList<String> classes
	
    

    AxiomsFromOnt(ontology, chosenReasoner){
	this.ontology = ontology
	this.chosenReasoner = chosenReasoner.toLowerCase()

	this.axioms_orig = new ArrayList<>()
	this.axioms_inf = new ArrayList<>()
	this.classes = new ArrayList<>()	
    }
    

    def getAxioms(){
	return this.axioms_orig
    }
    
    OWLOntologyManager outputManager = OWLManager.createOWLOntologyManager();
    OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
    
    public class SimpleShortFormProvider1 implements ShortFormProvider, Serializable {

	private final SimpleIRIShortFormProvider uriShortFormProvider = new SimpleIRIShortFormProvider();

	@Override
	public String getShortForm(OWLEntity entity) {
	    return '<'+entity.getIRI().toString()+'>';
	}
	public void dispose(){
	    ;
	}
    }

    

    def processOntology(){

	OWLDataFactory dataFactory = manager.getOWLDataFactory()
	OWLDataFactory fac = manager.getOWLDataFactory()

	ConsoleProgressMonitor progressMonitor = new ConsoleProgressMonitor()
	OWLReasonerConfiguration config = new SimpleConfiguration(progressMonitor)


//	ElkReasonerFactory f1 = new ElkReasonerFactory()
//	OWLReasoner reasoner = f1.createReasoner(this.ontology, config)
	//reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY)

//	List<InferredAxiomGenerator<? extends OWLAxiom>> gens = new ArrayList<InferredAxiomGenerator<? extends OWLAxiom>>();
//	gens.add(new InferredSubClassAxiomGenerator());
//	gens.add(new InferredEquivalentClassAxiomGenerator());
//	OWLOntology infOnt = outputManager.createOntology();


//	InferredOntologyGenerator iog = new InferredOntologyGenerator(reasoner,gens);
//	iog.fillOntology(outputManager.getOWLDataFactory(), infOnt);

	// Save the inferred ontology.
	//outputManager.saveOntology(infOnt,IRI.create((new File("inferredontologygo2.owl").toURI())));

	// Display Axioms
	OWLObjectRenderer renderer =new ManchesterOWLSyntaxOWLObjectRendererImpl ();
	renderer.setShortFormProvider(new SimpleShortFormProvider1());
//	int numaxiom1= infOnt.getAxiomCount();
//	Set<OWLClass> classes=infOnt.getClassesInSignature();
	
	//display original axioms
	//int numaxiom1= Ont.getAxiomCount();
	Set<OWLClass> classeso=this.ontology.getClassesInSignature();
	
	
	for (OWLClass classo : classeso){
	    
	    Set<OWLClassAxiom> ontoaxioms=this.ontology.getAxioms (classo);
	    for (OWLClassAxiom claxiom: ontoaxioms) {
		// classess=renderer.render(class1);
		String classaxiom=renderer.render (claxiom);
		//out1.println (classess);
		this.axioms_orig.add(classaxiom.replaceAll("\n"," ").replaceAll(","," "));
	    }
	}
	   

    }
}

package org.mowl.Parsers

// OWL API imports
import org.semanticweb.owlapi.model._
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.parameters.Imports


// Java imports
import collection.JavaConverters._

import org.mowl.Types._

trait AbstractParser{

  def ontology:OWLOntology
  def bidirectional:Boolean
  def transitiveClosure:String //Expect to support: "subclass", "relations", "full"(subclass and relations)

  val ontManager = OWLManager.createOWLOntologyManager()
  val dataFactory = ontManager.getOWLDataFactory()

  def parse = {

    val axioms = ontology.getAxioms()
    val imports = Imports.fromBoolean(false)

    val goClasses = ontology.getClassesInSignature(imports).asScala.toList
    println("INFO: Number of ontology classes: ${goClasses.length}")


    if (transitiveClosure != ""){
        getTransitiveClosure(goClasses)
    }

    val edges = goClasses.foldLeft(List[Edge]()){(acc, x) => acc ::: processOntClass(x)}

    edges.asJava

  }

  //Abstract methods
  def parseAxiom(ontClass: OWLClass, axiom: OWLClassAxiom): List[Edge]
  def getTransitiveClosure(goClasses:List[OWLClass])
  //////////////////////



  def processOntClass(ontClass: OWLClass): List[Edge] = {
    val axioms = ontology.getAxioms(ontClass).asScala.toList
    val edges = axioms.flatMap(parseAxiom(ontClass, _: OWLClassAxiom))
    edges
  }



 

}

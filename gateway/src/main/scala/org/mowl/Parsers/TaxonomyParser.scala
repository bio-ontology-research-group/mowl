package org.mowl.Parsers

// OWL API imports
import org.semanticweb.owlapi.model._
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.parameters.Imports
import uk.ac.manchester.cs.owl.owlapi._

import org.semanticweb.elk.owlapi.ElkReasonerFactory;

// Java imports
import java.io.File


import collection.JavaConverters._
import org.mowl.Types._


class TaxonomyParser(var ontology: OWLOntology, var bidirectional: Boolean = true, var transitiveClosure:String="") extends AbstractParser{

  def parseAxiom(go_class: OWLClass, axiom: OWLClassAxiom): List[Edge] = {
    val axiomType = axiom.getAxiomType().getName()
    axiomType match {
      case "SubClassOf" => {
	var ax = axiom.asInstanceOf[OWLSubClassOfAxiom]
	parseSubClassAxiom(ax.getSubClass.asInstanceOf[OWLClass], ax.getSuperClass)
      }

      case _ => Nil
    }
  }


  def parseSubClassAxiom(go_class: OWLClass, superClass: OWLClassExpression): List[Edge] = {

    val superClass_type = superClass.getClassExpressionType().getName()

    superClass_type match {

      case "Class" => {
	val dst = superClass.asInstanceOf[OWLClass]
        if (bidirectional){
	  new Edge(go_class, "subClassOf", dst) :: new Edge(dst, "superClassOf", go_class) :: Nil
        }else{
          new Edge(go_class, "subClassOf", dst) :: Nil
        }
      }
      case _ => Nil

    }

  }


  def getTransitiveClosure(goClasses:List[OWLClass]){

    if (transitiveClosure == "subclass"){
      val reasonerFactory = new ElkReasonerFactory();
      val reasoner = reasonerFactory.createReasoner(ontology);

      val superClasses = (cl:OWLClass) => (cl, reasoner.getSuperClasses(cl, false).getFlattened.asScala.toList)

      //aux function
      val transitiveAxioms = (tuple: (OWLClass, List[OWLClass])) => {
        val subclass = tuple._1
        val superClasses = tuple._2
        superClasses.map((sup) => new OWLSubClassOfAxiomImpl(subclass, sup, Nil.asJava))
      }

      //compose aux functions
      val newAxioms = goClasses flatMap (transitiveAxioms compose  superClasses)

      ontManager.addAxioms(ontology, newAxioms.toSet.asJava)
    }
  }


}

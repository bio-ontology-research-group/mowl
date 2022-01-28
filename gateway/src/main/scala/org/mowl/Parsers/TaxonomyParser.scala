package org.mowl.Parsers

// OWL API imports
import org.semanticweb.owlapi.model._
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.parameters.Imports
import uk.ac.manchester.cs.owl.owlapi._

// Java imports
import java.io.File


import collection.JavaConverters._
import org.mowl.Types._


class TaxonomyParser(var ontology: OWLOntology, var bidirectional_taxonomy: Boolean = false) extends AbstractParser{

  def parseAxiom(go_class: OWLClass, axiom: OWLClassAxiom): List[Triple] = {
    val axiomType = axiom.getAxiomType().getName()
    axiomType match {
      case "SubClassOf" => {
	val ax = axiom.asInstanceOf[OWLSubClassOfAxiom]
	parseSubClassAxiom(ax.getSubClass.asInstanceOf[OWLClass], ax.getSuperClass)
      }
      case _ => Nil
    }
  }


  def parseSubClassAxiom(go_class: OWLClass, superClass: OWLClassExpression): List[Triple] = {

    val superClass_type = superClass.getClassExpressionType().getName()

    superClass_type match {

      case "Class" => {
	val dst = superClass.asInstanceOf[OWLClass]
        if (bidirectional_taxonomy){
	  new Triple(go_class, "subClassOf", dst) :: new Triple(dst, "superClassOf", go_class) :: Nil
        }else{
          new Triple(go_class, "subClassOf", dst) :: Nil
        }
      }
      case _ => Nil

    }

  }

}

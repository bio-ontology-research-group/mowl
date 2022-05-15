package org.mowl.Projectors

// OWL API imports
import org.semanticweb.owlapi.model._
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.parameters.Imports
import uk.ac.manchester.cs.owl.owlapi._ 

// Java imports
import java.io.File
import java.util.{HashMap, ArrayList}

import collection.JavaConverters._
import org.mowl.Types._


class TaxonomyWithRelsProjector(
  var taxonomy: Boolean = false,
  var bidirectional_taxonomy: Boolean=false,
  var relations: ArrayList[String]) extends AbstractProjector{

  val relationsSc = relations.asScala
  if (!taxonomy) bidirectional_taxonomy = false

  def projectAxiom(ontClass: OWLClass, axiom: OWLClassAxiom): List[Triple] = {
    val axiomType = axiom.getAxiomType().getName()
    axiomType match {

      case "SubClassOf" => {
	var ax = axiom.asInstanceOf[OWLSubClassOfAxiom]
	projectSubClassAxiom(ax.getSubClass.asInstanceOf[OWLClass], ax.getSuperClass)
      }

      case _ => Nil
    }
  }


  def projectSubClassAxiom(ontClass: OWLClass, superClass: OWLClassExpression): List[Triple] = {

    val superClass_type = superClass.getClassExpressionType().getName()

    superClass_type match {

      case "ObjectSomeValuesFrom" => {

	val superClass_ = superClass.asInstanceOf[OWLObjectSomeValuesFrom]
                
	val (rel, dstClass) = projectQuantifiedExpression(Existential(superClass_))

        rel match {
          case r if (relationsSc contains rel) => {
	    val dstType = dstClass.getClassExpressionType().getName()
		    
	    dstType match {
	      case "Class" => {
	        val dst = dstClass.asInstanceOf[OWLClass]
                new Triple(ontClass, r, dst) :: Nil
              }
	      case _ => Nil
	    }
          }
          case _ => Nil
        }
      }

      case "Class" => {
	val dst = superClass.asInstanceOf[OWLClass]

        if (bidirectional_taxonomy){
	  new Triple(ontClass, "subclassOf", dst) :: new Triple(dst, "superclassOf", ontClass) :: Nil
        }
        else if (taxonomy){
          new Triple(ontClass, "subclassOf", dst) :: Nil
        }else Nil

      }
      case _ => Nil

    }

  }

    
  def projectQuantifiedExpression(expr: QuantifiedExpression) = {
        
    var relation = expr.getProperty.asInstanceOf[OWLObjectProperty]
    val rel = relation.getIRI().toString
    val dstClass = expr.getFiller

    (rel, dstClass)
        
  }

  // Abstract methods
  def projectAxiom(go_class: OWLClass, axiom: OWLClassAxiom, ontology: OWLOntology): List[Triple] = Nil

  

}

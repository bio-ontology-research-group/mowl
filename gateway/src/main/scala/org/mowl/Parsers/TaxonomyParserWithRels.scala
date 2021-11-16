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


class TaxonomyParserWithRels(var ontology: OWLOntology, var bidirectional_taxonomy: Boolean=false) extends AbstractParser{

  def parseAxiom(goClass: OWLClass, axiom: OWLClassAxiom): List[Edge] = {
    val axiomType = axiom.getAxiomType().getName()
    axiomType match {

      case "SubClassOf" => {
	var ax = axiom.asInstanceOf[OWLSubClassOfAxiom]
	parseSubClassAxiom(ax.getSubClass.asInstanceOf[OWLClass], ax.getSuperClass)
      }

      case _ => Nil
    }
  }


  def parseSubClassAxiom(goClass: OWLClass, superClass: OWLClassExpression): List[Edge] = {

    val superClass_type = superClass.getClassExpressionType().getName()

    superClass_type match {

      case "ObjectSomeValuesFrom" => {

	val superClass_ = superClass.asInstanceOf[OWLObjectSomeValuesFrom]
                
	val (rel, dstClass) = parseQuantifiedExpression(Existential(superClass_))

	val dstType = dstClass.getClassExpressionType().getName()
		    
	dstType match {
	  case "Class" => {
	    val dst = dstClass.asInstanceOf[OWLClass]

            if (bidirectional_taxonomy) {
	      new Edge(goClass, rel, dst) :: new Edge(dst, "INVERSE_OF_"+rel, goClass) :: Nil
            }else{
	      new Edge(goClass, rel, dst) :: Nil
	    }
          }
	  case _ => Nil
	}

      }

      case "Class" => {
	val dst = superClass.asInstanceOf[OWLClass]
        if (bidirectional_taxonomy){
	  new Edge(goClass, "subClassOf", dst) :: new Edge(dst, "superClassOf", goClass) :: Nil
        }else{
          new Edge(goClass, "subClassOf", dst) :: Nil
        }

      }
      case _ => Nil

    }

  }

    
  def parseQuantifiedExpression(expr: QuantifiedExpression) = {
        
    var relation = expr.getProperty.asInstanceOf[OWLObjectProperty]
    val rel = relation.getIRI().toString
    val dstClass = expr.getFiller

    (rel, dstClass)
        
  }


}

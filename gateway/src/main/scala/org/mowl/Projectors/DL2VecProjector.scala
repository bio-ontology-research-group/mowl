package org.mowl.Projectors

// OWL API imports
import org.semanticweb.owlapi.model._
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.parameters.Imports
import uk.ac.manchester.cs.owl.owlapi._

import org.semanticweb.owlapi.reasoner.InferenceType

import org.semanticweb.owlapi.util._
// Java imports
import java.io.File


import collection.JavaConverters._
import org.mowl.Types._
import org.mowl.Utils._

class DL2VecProjector(var bidirectional_taxonomy: Boolean = false) extends AbstractProjector{

  val quantityModifiers = List("ObjectSomeValuesFrom", "ObjectAllValuesFrom", "ObjectMaxCardinality", "ObjectMinCardinality")

  val collectors = List("ObjectIntersectionOf", "ObjectUnionOf")

  def projectAxiom(ontClass: OWLClass, axiom: OWLClassAxiom): List[Triple] = {

    val axiomType = axiom.getAxiomType().getName()

    axiomType match {

      case "SubClassOf" => {
	var ax = axiom.asInstanceOf[OWLSubClassOfAxiom]
	projectSubClassOrEquivAxiom(ax.getSubClass.asInstanceOf[OWLClass], ax.getSuperClass, "subclassOf")
      }
      case "EquivalentClasses" => {
	var ax = axiom.asInstanceOf[OWLEquivalentClassesAxiom].getClassExpressionsAsList.asScala
        val rightSide = ax.filter((x) => x != ontClass)
      	rightSide.toList.flatMap(projectSubClassOrEquivAxiom(ontClass, _:OWLClassExpression, "subclassOf"))
      }
      case _ => Nil
    }
  }

   def projectSubClassOrEquivAxiom(ontClass: OWLClass, superClass: OWLClassExpression, relName: String): List[Triple] = {
     var invRelName = ""

     if (relName == "subclassOf"){
       invRelName = "superclassOf"
     }else if(relName == "equivalentTo"){
       invRelName = relName
     }
     
    val superClassType = superClass.getClassExpressionType.getName

    superClassType match {

      case m if (quantityModifiers contains m) => {
        val superClass_ = lift2QuantifiedExpression(superClass)
        val (relations, dstClass) = projectQuantifiedExpression(superClass_, Nil)
        val dstClasses = splitClass(dstClass)

        for (
          rel <- relations;
          dst <- dstClasses.filter(_.getClassExpressionType.getName == "Class").map(_.asInstanceOf[OWLClass])
        ) yield new Triple(ontClass, rel, dst)
      }

      case c if (collectors contains c) => {
        val dstClasses = splitClass(superClass)
        dstClasses.flatMap(projectSubClassOrEquivAxiom(ontClass, _:OWLClassExpression, "subclassOf"))
      }

      case "Class" => {
	val dst = superClass.asInstanceOf[OWLClass]
        if (bidirectional_taxonomy){
	  new Triple(ontClass, relName, dst) :: new Triple(dst, invRelName, ontClass) :: Nil
        }else{
          new Triple(ontClass, relName, dst) :: Nil
        }
      }
      case _ => Nil
    }
   }

  def projectQuantifiedExpression(expr:QuantifiedExpression, relations:List[String]): (List[String], OWLClassExpression) = {
    val rel = expr.getProperty.asInstanceOf[OWLObjectProperty]
    val relName = rel.toString
    var filler = expr.getFiller

    defineQuantifiedExpression(filler) match {
      case Some(expr) => projectQuantifiedExpression(expr, relName::relations)
      case None => (relName::relations, filler)
    }
  }

  def splitClass(classExpr:OWLClassExpression): List[OWLClassExpression] = {
    val exprType = classExpr.getClassExpressionType.getName

    exprType match {
      case "Class" => classExpr.asInstanceOf[OWLClass] :: Nil
      case m if (quantityModifiers contains m) => classExpr :: Nil

      case "ObjectIntersectionOf" => {
        val classExprInt = classExpr.asInstanceOf[OWLObjectIntersectionOf]
        val operands = classExprInt.getOperands.asScala.toList
        operands.flatMap(splitClass(_: OWLClassExpression))
      }

      case "ObjectUnionOf" => {
        val classExprUnion = classExpr.asInstanceOf[OWLObjectUnionOf]
        val operands = classExprUnion.getOperands.asScala.toList
        operands.flatMap(splitClass(_: OWLClassExpression))
      }
      case _ => Nil
    }
  }

  // Abstract methods
  def projectAxiom(go_class: OWLClass, axiom: OWLClassAxiom, ontology: OWLOntology): List[Triple] = Nil
}

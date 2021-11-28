package org.mowl.Parsers

// OWL API imports
import org.semanticweb.owlapi.model._
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.parameters.Imports
import uk.ac.manchester.cs.owl.owlapi._

import org.semanticweb.owlapi.reasoner.InferenceType

import org.semanticweb.owlapi.util._
// Java imports
import java.io.File
import java.util

import collection.JavaConverters._
import org.mowl.Types._
import org.mowl.Utils._

class OWL2VecStarParser(
  var ontology: OWLOntology,
  var bidirectional_taxonomy: Boolean,
  var only_taxonomy: Boolean,
  var include_literals: Boolean,
  var avoid_properties: java.util.HashSet[String],
  var additional_preferred_labels_annotations: java.util.HashSet[String],
  var additional_synonyms_annotations: java.util.HashSet[String],
  var memory_reasoner: String = "10240"
) extends AbstractParser{


  avoid_properties = avoid_properties//.asScala
  additional_synonyms_annotations = additional_synonyms_annotations//.asScala
  additional_preferred_labels_annotations = additional_preferred_labels_annotations//.asScala


  override def parse = {

    val goClasses = ontology.getClassesInSignature().asScala.toList
    printf("INFO: Number of ontology classes: %d", goClasses.length)

    val edges = goClasses.foldLeft(List[Edge]()){(acc, x) => acc ::: processOntClass(x)}

    edges.asJava
  }


  // override def processOntClass(ontClass: OWLClass): List[Edge] = {
  //   val axioms = ontology.getAxioms(ontClass).asScala.toList
  //   val edges = axioms.flatMap(parseAxiom(ontClass, _: OWLClassAxiom))
  //   edges
  // }



  def parseAxiom(goClass: OWLClass, axiom: OWLClassAxiom): List[Edge] = {

    val axiomType = axiom.getAxiomType().getName()

    axiomType match {

      case "SubClassOf" => {
	var ax = axiom.asInstanceOf[OWLSubClassOfAxiom]
	parseSubClassOrEquivAxiom(ax.getSubClass.asInstanceOf[OWLClass], ax.getSuperClass)
      }
      case "EquivalentClasses" => {
	var ax = axiom.asInstanceOf[OWLEquivalentClassesAxiom].getClassExpressionsAsList.asScala
        assert(goClass == ax.head)
        val rightSide = ax.tail
	parseSubClassOrEquivAxiom(goClass, new OWLObjectIntersectionOfImpl(rightSide.toSet.asJava))
      }

      case _ => Nil
    }
  }


   def parseSubClassOrEquivAxiom(goClass: OWLClass, superClass: OWLClassExpression): List[Edge] = {

     val quantityModifiers = List("ObjectSomeValuesFrom", "ObjectAllValuesFrom", "ObjectMaxCardinality", "ObjectMinCardinality")

     val superClassType = superClass.getClassExpressionType.getName

    superClassType match {

      case m if (quantityModifiers contains m) => {

	val superClass_ = defineQuantifiedExpression(superClass)

        parseQuantifiedExpression(superClass_) match {

          case Some((rel, dstClass)) => {
            val dstClasses = splitClass(dstClass)

            for (
              dst <- dstClasses
            ) yield new Edge(goClass, rel, dst)
          }

          case None => Nil
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

  def parseQuantifiedExpression(expr:QuantifiedExpression): Option[(String, OWLClassExpression)] = {

    val rel = expr.getProperty.asInstanceOf[OWLObjectProperty]
    val relName = rel.getIRI.toString
    val filler = expr.getFiller

    val fillerType = filler.getClassExpressionType.getName

    fillerType match {
      case "Class" => Some((relName, filler.asInstanceOf[OWLClass]))

      case _ => None
    }
  }

  def splitClass(classExpr:OWLClassExpression): List[OWLClass] = {
    val exprType = classExpr.getClassExpressionType.getName

    exprType match {
      case "Class" => classExpr.asInstanceOf[OWLClass] :: Nil

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

}

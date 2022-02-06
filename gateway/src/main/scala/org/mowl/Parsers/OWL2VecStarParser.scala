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


import collection.JavaConverters._
import org.mowl.Types._


class OWL2VecStarParser(
  var ontology: OWLOntology,
  var bidirectional_taxonomy: Boolean = false,
  var only_taxonomy: Boolean = false,
  var include_literals: Boolean = true
  // var avoid_properties: Set[String] = Set(),
  // var additional_preferred_labels_annotations: Set[String] = Set(),
  // var additional_synonyms_annotations: Set[String] = Set(),
  // var memory_reasoner: String = "10240"
) extends AbstractParser{

  override def parse = {

    val goClasses = ontology.getClassesInSignature().asScala.toList
    printf("INFO: Number of ontology classes: %d", goClasses.length)

    val edges = goClasses.foldLeft(List[Triple]()){(acc, x) => acc ::: processOntClass(x)}

    edges.asJava
  }

  def parseAxiom(goClass: OWLClass, axiom: OWLClassAxiom): List[Triple] = {

    val axiomType = axiom.getAxiomType().getName()



    axiomType match {

      case "SubClassOf" => {
	var ax = axiom.asInstanceOf[OWLSubClassOfAxiom]
	parseSubClassOrEquivAxiom(ax.getSubClass.asInstanceOf[OWLClass], ax.getSuperClass, "subClassOf")
      }
      case "EquivalentClasses" => {
	  var ax = axiom.asInstanceOf[OWLEquivalentClassesAxiom].getClassExpressionsAsList.asScala
          assert(goClass == ax.head)
          val rightSide = ax.tail
	  parseSubClassOrEquivAxiom(goClass, new OWLObjectIntersectionOfImpl(rightSide.toSet.asJava), "equivalentTo")
      }

      

      case _ => Nil
    }
  }



   def parseSubClassOrEquivAxiom(goClass: OWLClass, superClass: OWLClassExpression, relName: String): List[Triple] = {
     var invRelName = ""

     if (relName == "subClassOf"){
       invRelName = "superClassOf"
     }else if(relName == "equivalentTo"){
       invRelName = relName
     }
     

    val superClassType = superClass.getClassExpressionType.getName

    superClassType match {

      case "ObjectSomeValuesFrom" => {

        if (relName == "subClassOf" && !only_taxonomy){

	  val superClass_ = superClass.asInstanceOf[OWLObjectSomeValuesFrom]          
	  val (relations, dstClass) = parseQuantifiedExpression(Existential(superClass_), Nil)
          val dstClasses = splitClass(dstClass)

          for (
            rel <- relations;
            dst <- dstClasses
          ) yield new Triple(goClass, rel, dst)

        }else Nil
      }

      case "Class" => {
	val dst = superClass.asInstanceOf[OWLClass]
        if (bidirectional_taxonomy){
	  new Triple(goClass, relName, dst) :: new Triple(dst, invRelName, goClass) :: Nil
        }else{
          new Triple(goClass, relName, dst) :: Nil
        }

      }
      case _ => Nil

    }

   }

  def parseQuantifiedExpression(expr:QuantifiedExpression, relations:List[String]): (List[String], OWLClassExpression) = {
    val rel = expr.getProperty.asInstanceOf[OWLObjectProperty]
    val relName = rel.getIRI.toString
    var filler = expr.getFiller

    val fillerType = filler.getClassExpressionType.getName

    fillerType match {
      case "ObjectSomeValuesFrom" => {
        val fillerSome = filler.asInstanceOf[OWLObjectSomeValuesFrom]
        parseQuantifiedExpression(Existential(fillerSome), relName::relations)
      }
      case "ObjectAllValuesFrom" => {
        val fillerAll = filler.asInstanceOf[OWLObjectAllValuesFrom]
        parseQuantifiedExpression(Universal(fillerAll), relName::relations)
      }
      case "ObjectMinCardinality" => {
        val fillerMin = filler.asInstanceOf[OWLObjectMinCardinality]
        parseQuantifiedExpression(MinCardinality(fillerMin), relName::relations)
      }
      case "ObjectMaxCardinality" => {
        val fillerMax = filler.asInstanceOf[OWLObjectMaxCardinality]
        parseQuantifiedExpression(MaxCardinality(fillerMax), relName::relations)
      }
      case _ => {
        (relName::relations, filler)
      }
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

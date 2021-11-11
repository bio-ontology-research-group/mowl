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


class DL2VecParser(var ontology: OWLOntology, var bidirectional: Boolean = true, var transitiveClosure: String = "") extends AbstractParser{

  var relCounter = 0


  override def parse = {
    println("INFO: Overrided parse function")

    val imports = Imports.fromBoolean(true)

    val goClasses = ontology.getClassesInSignature().asScala.toList
    println("INFO: Number of ontology classes: ${goClasses.length}")

    val edges = goClasses.foldLeft(List[Edge]()){(acc, x) => acc ::: processOntClass(x)}

    edges.asJava
  }

  


  def parseAxiom(goClass: OWLClass, axiom: OWLClassAxiom): List[Edge] = {

    if (goClass.getIRI.toString == "http://purl.obolibrary.org/obo/GO_0002278"){
      println(axiom)}
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



   def parseSubClassOrEquivAxiom(goClass: OWLClass, superClass: OWLClassExpression, relName: String): List[Edge] = {
     var invRelName = ""

     if (relName == "subClassOf"){
       invRelName = "superClassOf"
     }else if(relName == "equivalentTo"){
       invRelName = relName
     }
     

    val superClassType = superClass.getClassExpressionType.getName

    superClassType match {

      case "ObjectSomeValuesFrom" => {

	val superClass_ = superClass.asInstanceOf[OWLObjectSomeValuesFrom]
                
	val (relations, dstClass) = parseQuantifiedExpression(Existential(superClass_), Nil)

        val dstClasses = splitClass(dstClass)


        for (
          rel <- relations;
          dst <- dstClasses
        ) yield new Edge(goClass, rel, dst)

      }

      case "Class" => {
	val dst = superClass.asInstanceOf[OWLClass]
        if (bidirectional){
	  new Edge(goClass, relName, dst) :: new Edge(dst, invRelName, goClass) :: Nil
        }else{
          new Edge(goClass, relName, dst) :: Nil
        }

      }
      case _ => Nil

    }

   }

  def parseQuantifiedExpression(expr:QuantifiedExpression, relations:List[String]): (List[String], OWLClassExpression) = {
    val rel = expr.getProperty.asInstanceOf[OWLObjectProperty]
    val relName = getRelationName(rel)
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

  def getRelationName(relation: OWLObjectProperty) = {
        
    val rel_annots = ontology.getAnnotationAssertionAxioms(relation.getIRI()).asScala.toList

    val rel = rel_annots find (x => x.getProperty() == dataFactory.getRDFSLabel()) match {
      case Some(r) => r.getValue().toString.replace("\"", "").replace(" ", "_")
      case None => {
        relCounter = relCounter + 1
        "rel" + (relCounter)
      }
    }

    rel
  }


  def getTransitiveClosure(goClasses:List[OWLClass]){}

}

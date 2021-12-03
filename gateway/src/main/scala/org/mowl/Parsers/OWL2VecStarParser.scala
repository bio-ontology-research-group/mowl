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


  val inverseRelations = scala.collection.mutable.Map[String, Option[String]]()

  override def parse = {

    var edgesFromObjectProperties = List[Edge]()

    val goClasses = ontology.getClassesInSignature().asScala.toList
    printf("INFO: Number of ontology classes: %d", goClasses.length)
    val edgesFromClasses = goClasses.foldLeft(List[Edge]()){(acc, x) => acc ::: processOntClass(x)}

    if (!only_taxonomy) { //TODO: Check if this condition is doing something

      val objectProperties = ontology.getObjectPropertiesInSignature().asScala.toList.filter(o => !(avoid_properties contains o))

      edgesFromObjectProperties = objectProperties.foldLeft(List[Edge]()){(acc, x) => acc ::: processObjectProperty(x)}

    }

 
    (edgesFromClasses ::: edgesFromObjectProperties).asJava
  }





  // CLASSES PROCESSING

  override def processOntClass(ontClass: OWLClass): List[Edge] = {

    var annotationEdges = List[Edge]()

    if (include_literals){ //ANNOTATION PROCESSSING
      val annotationAxioms = ontology.getAnnotationAssertionAxioms(ontClass.getIRI).asScala.toList
      annotationEdges = annotationAxioms.map(annotationAxiom2Edge).flatten
    }

    val axioms = ontology.getAxioms(ontClass).asScala.toList
    val edges = axioms.flatMap(parseAxiom(ontClass, _: OWLClassAxiom))
    edges ::: annotationEdges
  }



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

      case m if (quantityModifiers contains m) && !only_taxonomy => {

	val superClass_ = lift2QuantifiedExpression(superClass)

        parseQuantifiedExpression(superClass_) match {

          case Some((rel, Some(inverseRel), dstClass)) => {
            val dstClasses = splitClass(dstClass)

            val outputEdges = for (dst <- dstClasses)
            yield List(new Edge(goClass, rel, dst), new Edge(dst, inverseRel, goClass))

            outputEdges.flatten
          }

          case Some((rel, None, dstClass)) => {
            val dstClasses = splitClass(dstClass)

            for (dst <- dstClasses) yield new Edge(goClass, rel, dst)

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

  def parseQuantifiedExpression(expr:QuantifiedExpression): Option[(String, Option[String], OWLClassExpression)] = {

    val rel = expr.getProperty.asInstanceOf[OWLObjectProperty]

    val (relName, inverseRelName) = getRelationInverseNames(rel)
    
    val filler = expr.getFiller

    val fillerType = filler.getClassExpressionType.getName

    fillerType match {
      case "Class" => Some((relName, inverseRelName, filler.asInstanceOf[OWLClass]))

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

  ////////////////////////////////////////////////////
  ////////////////////////////////////////////////////

  //OBJECT PROPERTIES PROCESSING

  def processObjectProperty(property: OWLObjectProperty): List[Edge] = {

    println(ontology.getObjectPropertyDomainAxioms(property))

    Nil

  }

  def getRelationInverseNames(relation: OWLObjectProperty): (String, Option[String]) = {
    val relName = relation.getIRI.toString

    if (inverseRelations.contains(relName)){
      (relName, inverseRelations(relName))
    }else{

      val rels = ontology.getInverseObjectPropertyAxioms(relation).asScala.toList

      if (!rels.isEmpty){
        println(relation, rels.head.getFirstProperty, rels.head.getSecondProperty)
        val inverseRelName = stripBracket(rels.head.getSecondProperty) //TODO: remove head and modify the code to deal with lists
        inverseRelations += (relName -> Some(inverseRelName))
        (relName, Some(inverseRelName))
      }else{
        inverseRelations += (relName -> None)
        (relName, None)
      }
    }
  }

  def stripBracket(value: OWLObjectPropertyExpression) = {
    val valueStr = value.toString

    valueStr.head match {
      case '<' => valueStr.tail.init
      case _ => valueStr
    }

  }

  //////////////////////////////////////////////////
  //////////////////////////////////////////////////

  //ANNOTATION PROPERTIES PROCESSING


  val excludedAnnotationProperties = List("rdfs:comment", "http://www.w3.org/2000/01/rdf-schema#comment")

  def annotationAxiom2Edge(annotationAxiom: OWLAnnotationAssertionAxiom): Option[Edge] = {

    val property = annotationAxiom.getProperty.toStringID.toString

    property match {

      case m if excludedAnnotationProperties contains m => None
      case _ => {
        val subject = annotationAxiom.getSubject
        val value = annotationAxiom.getValue
        
        Some(new Edge(subject, property, stripValue(value)))
      }
    }
  }

  def stripValue(value: OWLAnnotationValue) = {
    val valueStr = value.toString

    valueStr.head match {
      case '"' => valueStr.tail.init
      case _ => valueStr
    }

  }
}

package org.mowl.Projectors

// OWL API imports
import org.semanticweb.owlapi.model._

// Java imports
import collection.JavaConverters._
import org.mowl.Types._
import org.mowl.Utils._


/**
  * Projector for gene--disease association graphs.
  *
  * Follows the OWL2Vec* projection rules and adds one rule: existential
  * restrictions whose filler is not an atomic class are unfolded recursively,
  * descending through ObjectIntersectionOf and nested ObjectSomeValuesFrom
  * until a named class is reached. A single edge is emitted whose relation is
  * the composition of the roles traversed.
  *
  * The unfolding is scoped by IRI prefix on both ends: only classes matching
  * source_prefixes are used as sources, and only classes matching
  * target_prefixes as targets. An empty prefix list matches every class.
  */
class GDAProjector(
  bidirectional_taxonomy: Boolean,
  only_taxonomy: Boolean,
  include_literals: Boolean,
  source_prefixes: java.util.List[String],
  target_prefixes: java.util.List[String]
) extends OWL2VecStarProjector(bidirectional_taxonomy, only_taxonomy, include_literals) {

  val composedRelationNamespace = "http://mowl.borg/"

  private val sourcePrefixes = source_prefixes.asScala.toList
  private val targetPrefixes = target_prefixes.asScala.toList

  private def hasPrefix(iri: String, prefixes: List[String]): Boolean =
    prefixes.isEmpty || prefixes.exists(iri.startsWith(_))

  /**
    * Falls back to the nested unfolding when the OWL2Vec* rules produce no
    * edge, which happens exactly when the filler is not an atomic class.
    */
  override def processSubClassAxiomComplexSuperClass(
    subClass: OWLClassExpression,
    superClass: OWLClassExpression,
    ontology: OWLOntology): List[Triple] = {

    val baseTriples = super.processSubClassAxiomComplexSuperClass(subClass, superClass, ontology)

    if (baseTriples.nonEmpty || only_taxonomy) {
      baseTriples
    } else {
      subClass match {
        case subClass_ : OWLClass if hasPrefix(subClass_.toStringID, sourcePrefixes) =>
          processNestedSomeValuesFrom(subClass_, superClass)
        case _ => Nil
      }
    }
  }

  def processNestedSomeValuesFrom(subClass: OWLClass, superClass: OWLClassExpression): List[Triple] = {
    superClass.getClassExpressionType.getName match {
      case "ObjectSomeValuesFrom" => {
        val outerExpr = lift2QuantifiedExpression(superClass)
        val outerRelLocal = localName(outerExpr.getProperty.getNamedProperty.toStringID)
        extractFromExpression(subClass, outerExpr.getFiller, List(outerRelLocal))
      }
      case _ => Nil
    }
  }

  /**
    * Walks the filler, accumulating the local name of every role traversed.
    * Emits one edge per named class reached that matches target_prefixes.
    */
  def extractFromExpression(
    subClass: OWLClass,
    expr: OWLClassExpression,
    relPath: List[String]): List[Triple] = {

    expr.getClassExpressionType.getName match {
      case "Class" => {
        val dst = expr.asInstanceOf[OWLClass]
        if (hasPrefix(dst.toStringID, targetPrefixes)) {
          val combinedRel = composedRelationNamespace + relPath.mkString("_")
          new Triple(subClass, combinedRel, dst) :: Nil
        } else Nil
      }
      case "ObjectIntersectionOf" => {
        val operands = expr.asInstanceOf[OWLObjectIntersectionOf].getOperands.asScala.toList
        operands.flatMap(extractFromExpression(subClass, _, relPath))
      }
      case "ObjectSomeValuesFrom" => {
        val someExpr = lift2QuantifiedExpression(expr)
        val relLocal = localName(someExpr.getProperty.getNamedProperty.toStringID)
        extractFromExpression(subClass, someExpr.getFiller, relPath :+ relLocal)
      }
      case _ => Nil
    }
  }

  private def localName(iri: String): String = iri.split("/").last
}

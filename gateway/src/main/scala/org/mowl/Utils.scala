package org.mowl

import org.semanticweb.owlapi.model._
import org.mowl.Types._

object Utils {

  def defineQuantifiedExpression(classExpression: OWLClassExpression): QuantifiedExpression = {

    val expressionType = classExpression.getClassExpressionType().getName()


    expressionType match {
      case "ObjectSomeValuesFrom" => Existential(classExpression.asInstanceOf[OWLObjectSomeValuesFrom])

      case "ObjectAllValuesFrom" => Universal(classExpression.asInstanceOf[OWLObjectAllValuesFrom])

      case "ObjectMinCardinality" => MinCardinality(classExpression.asInstanceOf[OWLObjectMinCardinality])

      case "ObjectMaxCardinality" => MaxCardinality(classExpression.asInstanceOf[OWLObjectMaxCardinality])
    }
  }
}

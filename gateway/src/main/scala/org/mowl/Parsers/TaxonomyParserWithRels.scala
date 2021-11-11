package org.mowl.Parsers

// OWL API imports
import org.semanticweb.owlapi.model._
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.parameters.Imports
import uk.ac.manchester.cs.owl.owlapi._ 

import org.semanticweb.elk.owlapi.ElkReasonerFactory;


// Java imports
import java.io.File


import collection.JavaConverters._
import org.mowl.Types._


class TaxonomyParserWithRels(var ontology: OWLOntology, var bidirectional: Boolean=false, var transitiveClosure: String="") extends AbstractParser{

  var relCounter = 0

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

            if (bidirectional) {
	      new Edge(goClass, rel, dst) :: new Edge(dst, "inv_"+rel, goClass) :: Nil
            }else{
	      new Edge(goClass, rel, dst) :: Nil
	    }
          }
	  case _ => Nil
	}

      }

      case "Class" => {
	val dst = superClass.asInstanceOf[OWLClass]
        if (bidirectional){
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

    val rel = getRelationName(relation)

    val dstClass = expr.getFiller

    (rel, dstClass)
        
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

    ///////////////////////////////////////


  def getTransitiveClosure(goClasses:List[OWLClass]){

    val reasonerFactory = new ElkReasonerFactory();
    val reasoner = reasonerFactory.createReasoner(ontology);

    //aux axioms
    val superClasses = (cl:OWLClass) => (cl, reasoner.getSuperClasses(cl, false).getFlattened.asScala.toList)

    //aux function
    val transitiveAxioms = (tuple: (OWLClass, List[OWLClass])) => {
      val subclass = tuple._1
      val superClasses = tuple._2
      superClasses.map((sup) => new OWLSubClassOfAxiomImpl(subclass, sup, Nil.asJava))
    }

    transitiveClosure match {

      case "subclass" => {

        //compose aux functions
        val newAxioms = goClasses flatMap (transitiveAxioms compose  superClasses)

        ontManager.addAxioms(ontology, newAxioms.toSet.asJava)
      }

      case "relations" => {
        //TODO
      }

      case "full" => {
        val newAxioms = goClasses flatMap (transitiveAxioms compose  superClasses)

        //TODO

        ontManager.addAxioms(ontology, newAxioms.toSet.asJava)


      }


    }

  }

}

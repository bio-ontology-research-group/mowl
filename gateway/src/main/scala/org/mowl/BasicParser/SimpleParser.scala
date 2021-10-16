package org.mowl.BasicParser

// OWL API imports
import org.semanticweb.owlapi.model._
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.parameters.Imports
import uk.ac.manchester.cs.owl.owlapi._ //OWLObjectSomeValuesFromImpl


import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.owlapi.reasoner.OWLReasonerFactory;
import org.semanticweb.HermiT.ReasonerFactory;

// Java imports
import java.io.File


import collection.JavaConverters._

import org.mowl.Types._


class SimpleParser(var ontology: OWLOntology, var subclass: Boolean = true, var relations: Boolean=false) {

  private val ont_manager = OWLManager.createOWLOntologyManager()
  private val data_factory = ont_manager.getOWLDataFactory()

  var rel_counter = 0

	
  def parse = {
           
    val axioms = ontology.getAxioms()
    val imports = Imports.fromBoolean(true)

    val go_classes = ontology.getClassesInSignature(imports).asScala.toList


    val transitive_closure = false
    if (transitive_closure) {

      val reasonerFactory = new ReasonerFactory();
      val reasoner = reasonerFactory.createReasoner(ontology);

      val superClasses = (cl:OWLClass) => (cl, reasoner.getSuperClasses(cl, false).getFlattened.asScala.toList)
      val transitiveAxioms = (tuple: (OWLClass, List[OWLClass])) => {
        val subclass = tuple._1
        val superClasses = tuple._2
        superClasses.map((sup) => new OWLSubClassOfAxiomImpl(subclass, sup, Nil.asJava))
      }
      val newAxioms = go_classes flatMap (transitiveAxioms compose  superClasses)

      ontology.addAxioms(newAxioms.asJava)

      // go_classes.foreach{
      //   (cl:OWLClass) => reasoner.getSuperClasses(cl, false).asScala.toList.foreach{
      //     (sup:OWLClass) => ontology.addAxiom(new OWLSubClassOfAxiomImpl(cl, sup, Nil.asJava))
      //   }
      // }
    }

    println(s"INFO: Number of GO classes: ${go_classes.length}")
       
        
    val edges = go_classes.foldLeft(List[Edge]()){(acc, x) => acc ::: processGOClass(x)}

    val nodes = getNodes(edges)

    (edges).asJava
  }
      
  def processGOClass(go_class: OWLClass): List[Edge] = {
    val axioms = ontology.getAxioms(go_class).asScala.toList

    val edges = axioms.flatMap(parseAxiom(go_class, _: OWLClassAxiom))
    edges.flatten
  }

  def parseAxiom(go_class: OWLClass, axiom: OWLClassAxiom): List[Option[Edge]] = {
    val axiomType = axiom.getAxiomType().getName()
    axiomType match {
      case "SubClassOf" => {
	var ax = axiom.asInstanceOf[OWLSubClassOfAxiom]
	parseSubClassAxiom(ax.getSubClass.asInstanceOf[OWLClass], ax.getSuperClass)
      }

      case _ => Nil
    }
  }


  def parseSubClassAxiom(go_class: OWLClass, superClass: OWLClassExpression): List[Option[Edge]] = {

    val superClass_type = superClass.getClassExpressionType().getName()

    superClass_type match {

      case "ObjectSomeValuesFrom" => {
	if (relations) {
	  val superClass_ = superClass.asInstanceOf[OWLObjectSomeValuesFrom]
                
	  val (rel, dst_class) = parseQuantifiedExpression(Existential(superClass_))

	  val dst_type = dst_class.getClassExpressionType().getName()
		    
	  dst_type match {
	    case "Class" => {
	      val dst = dst_class.asInstanceOf[OWLClass]
	      Some (new Edge(go_class, rel, dst)) :: Nil
	    }
	    case _ => Nil
	  }
	}else{
	  Nil
	}

      }

      case "Class" => {
	val dst = superClass.asInstanceOf[OWLClass]
	Some(new Edge(go_class, "subClassOf", dst)) :: Some(new Edge(dst, "superClassOf", go_class)) :: Nil
      }
      case _ => Nil

    }

  }

    
  def parseQuantifiedExpression(expr: QuantifiedExpression, inverse: Boolean = false) = {
        
    var relation = expr.getProperty.asInstanceOf[OWLObjectProperty]

    val rel = getRelationName(relation, inverse)

    val dst_class = expr.getFiller

    (rel, dst_class)
        
  }

  def getRelationName(relation: OWLObjectProperty, inverse: Boolean = false) = {
        
    var relat = relation
    if (inverse) {
      val inv_relation = relation.getInverseProperty
      if (!inv_relation.isAnonymous){
        relat = inv_relation.asOWLObjectProperty
      }

    }

    val rel_annots = ontology.getAnnotationAssertionAxioms(relat.getIRI()).asScala.toList

    val rel = rel_annots find (x => x.getProperty() == data_factory.getRDFSLabel()) match {
      case Some(r) => r.getValue().toString.replace("\"", "").replace(" ", "_")
      case None => {
        rel_counter = rel_counter + 1
        "rel" + (rel_counter)
      }
    }

    rel
  }

    ///////////////////////////////////////

}

package org.mowl.Projectors

// OWL API imports
import org.semanticweb.owlapi.model._
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.parameters.Imports
import uk.ac.manchester.cs.owl.owlapi._

import org.semanticweb.owlapi.reasoner.InferenceType

import org.semanticweb.owlapi.util._
import org.semanticweb.owlapi.search._
// Java imports
import java.io.File
import java.util
import scala.collection.mutable.ListBuffer
import collection.JavaConverters._
import org.mowl.Types._
import org.mowl.Utils._

class OWL2VecStarProjector(
  var bidirectional_taxonomy: Boolean,
  var only_taxonomy: Boolean,
  var include_literals: Boolean
  // var avoid_properties: java.util.HashSet[String],
  // var additional_preferred_labels_annotations: java.util.HashSet[String],
  // var additional_synonyms_annotations: java.util.HashSet[String],
  // var memory_reasoner: String = "10240"
) extends AbstractProjector{

  val inverseRelations = scala.collection.mutable.Map[String, Option[String]]()
  val searcher = new EntitySearcher()

  override def project(ontology: OWLOntology) = {

    var edgesFromObjectProperties = List[Triple]()
    val axioms = ontology.getAxioms(imports).asScala.toList

    var subclassOfAxioms = ListBuffer[OWLSubClassOfAxiom]()
    var equivalenceAxioms = ListBuffer[OWLEquivalentClassesAxiom]()
    var annotationAxioms = ListBuffer[OWLAnnotationAssertionAxiom]()
    var otherAxioms = ListBuffer[OWLAxiom]()

    for (axiom <- axioms){

      axiom.getAxiomType.getName match {
        case "SubClassOf" => subclassOfAxioms += axiom.asInstanceOf[OWLSubClassOfAxiom]
        case "AnnotationAssertion" => {
          if (include_literals)
          annotationAxioms += axiom.asInstanceOf[OWLAnnotationAssertionAxiom]
        }
        case "EquivalentClasses" => equivalenceAxioms += axiom.asInstanceOf[OWLEquivalentClassesAxiom]
        case _ => {
          //println(axiom)
          otherAxioms += axiom
        }
      }
    }

    val subclassOfTriples = subclassOfAxioms.flatMap(x => processSubClassAxiom(x.getSubClass, x.getSuperClass, ontology))
    val equivalenceTriples = equivalenceAxioms.flatMap(
      x => {
        val subClass::superClass::rest= x.getClassExpressionsAsList.asScala.toList
        superClass.getClassExpressionType.getName match{
          case "ObjectIntersectionOf" => superClass.asInstanceOf[OWLObjectIntersectionOf].getOperands.asScala.toList.flatMap(processSubClassAxiom(subClass, _, ontology))
          case _ => Nil
        }
      }
    )
    val annotationTriples = annotationAxioms.map(processAnnotationAxiom(_)).flatten
    (subclassOfTriples.toList ::: equivalenceTriples.toList ::: annotationTriples.toList).asJava
  }

  // CLASSES PROCESSING

  override def processOntClass(ontClass: OWLClass, ontology: OWLOntology): List[Triple] = {

    var annotationEdges = List[Triple]()

    if (include_literals){ //ANNOTATION PROCESSSING
      val annotProperties = ontClass.getAnnotationPropertiesInSignature.asScala.toList
      val annotationAxioms = ontology.getAnnotationAssertionAxioms(ontClass.getIRI).asScala.toList
      annotationEdges = annotationAxioms.map(annotationAxiom2Edge).flatten
    }

    val axioms = ontology.getAxioms(ontClass, imports).asScala.toList
    val edges = axioms.flatMap(projectAxiom(ontClass, _: OWLClassAxiom))
    edges ::: annotationEdges
  }



  def projectAxiom(ontClass: OWLClass, axiom: OWLClassAxiom, ontology: OWLOntology): List[Triple] = {

    val axiomType = axiom.getAxiomType().getName()

    axiomType match {
      case "SubClassOf" => {
	var ax = axiom.asInstanceOf[OWLSubClassOfAxiom]
	projectSubClassOrEquivAxiom(ax.getSubClass.asInstanceOf[OWLClass], ax.getSuperClass, ontology)
      }
      case "EquivalentClasses" => {
	var ax = axiom.asInstanceOf[OWLEquivalentClassesAxiom].getClassExpressionsAsList.asScala
        assert(ontClass == ax.head)
        val rightSide = ax.tail
	projectSubClassOrEquivAxiom(ontClass, new OWLObjectIntersectionOfImpl(rightSide.toSet.asJava), ontology)
      }
      case _ => Nil
    }
  }

  def processSubClassAxiom(subClass: OWLClassExpression, superClass: OWLClassExpression, ontology: OWLOntology): List[Triple] = {

    val firstCase = processSubClassAxiomComplexSubClass(subClass, superClass, ontology)

    if (firstCase == Nil){
      processSubClassAxiomComplexSuperClass(subClass, superClass, ontology)
    }else{
      firstCase
    }
  }

  def processSubClassAxiomComplexSubClass(subClass: OWLClassExpression, superClass: OWLClassExpression, ontology: OWLOntology): List[Triple] = {
    // When subclass is complex, superclass must be atomic

    val quantityModifiers = List("ObjectSomeValuesFrom", "ObjectAllValuesFrom", "ObjectMaxCardinality", "ObjectMinCardinality")
    val superClassType = superClass.getClassExpressionType.getName

    if (superClassType != "Class") {
      Nil
    }else{

      val superClass_ = superClass.asInstanceOf[OWLClass]
      val subClassType = subClass.getClassExpressionType.getName

      subClassType match {

        case m if (quantityModifiers contains m) && !only_taxonomy => {

	  val subClass_ = lift2QuantifiedExpression(subClass)
          projectQuantifiedExpression(subClass_, ontology) match {

            case Some((rel, Some(inverseRel), dstClass)) => {
              val dstClasses = splitClass(dstClass)

              val outputEdges = for (dst <- dstClasses)
              yield List(new Triple(superClass_, rel, dst), new Triple(dst, inverseRel, superClass_))
              outputEdges.flatten
            }

            case Some((rel, None, dstClass)) => {
              val dstClasses = splitClass(dstClass)
              for (dst <- dstClasses) yield new Triple(superClass_, rel, dst)
            }
            case None => Nil
          }
        }
        case _ => Nil
      }
    }
  }


  def processSubClassAxiomComplexSuperClass(subClass: OWLClassExpression, superClass: OWLClassExpression, ontology: OWLOntology): List[Triple] = {

    // When superclass is complex, subclass must be atomic

    val subClassType = subClass.getClassExpressionType.getName
    if (subClassType != "Class"){
      Nil
    }else{
      val subClass_  = subClass.asInstanceOf[OWLClass]
      val quantityModifiers = List("ObjectSomeValuesFrom", "ObjectAllValuesFrom", "ObjectMaxCardinality", "ObjectMinCardinality")
      val superClassType = superClass.getClassExpressionType.getName

      superClassType match {

        case m if (quantityModifiers contains m) && !only_taxonomy => {
	  val superClass_ = lift2QuantifiedExpression(superClass)
          projectQuantifiedExpression(superClass_, ontology) match {

            case Some((rel, Some(inverseRel), dstClass)) => {
              val dstClasses = splitClass(dstClass)
              val outputEdges = for (dst <- dstClasses)
              yield List(new Triple(subClass_, rel, dst), new Triple(dst, inverseRel, subClass_))
              outputEdges.flatten
            }
            case Some((rel, None, dstClass)) => {
              val dstClasses = splitClass(dstClass)
              for (dst <- dstClasses) yield new Triple(subClass_, rel, dst)
            }
            case None => Nil
          }
        }
        case "Class" => {
	  val dst = superClass.asInstanceOf[OWLClass]
          if (bidirectional_taxonomy){
	    new Triple(subClass_, "http://subclassof", dst) :: new Triple(dst, "http://superclassof", subClass_) :: Nil
          }else{
            new Triple(subClass_, "http://subclassof", dst) :: Nil
          }
        }
        case _ => Nil
      }
    }
  }



  def processAnnotationAxiom(axiom: OWLAnnotationAssertionAxiom): Option[Triple]= {
    val property = stripValue(axiom.getProperty.toString)

    property match {
      case m if (lexicalAnnotationURIs contains m) => {
        val subject = axiom.getSubject.toString
        val value = axiom.getValue
       
        val valueStr = value.isLiteral match {
          case true => {
            val datatype = value.asLiteral.get.getDatatype

            if (datatype.isString) value.asInstanceOf[OWLLiteralImplString].getLiteral
            else if(datatype.isRDFPlainLiteral) value.asInstanceOf[OWLLiteralImplPlain].getLiteral
            else {
              println("Warning: datatype not detected: ", datatype)
              stripValue(axiom.getValue.toString)
            } 
          }
          case false => stripValue(axiom.getValue.toString)
        }
        Some(new Triple(subject, m, valueStr))
      }
      case _ => {
        //println("C ",property)
        None
      }
    }
  }

  def projectSubClassOrEquivAxiom(ontClass: OWLClass, superClass: OWLClassExpression, ontology: OWLOntology): List[Triple] = {

    val quantityModifiers = List("ObjectSomeValuesFrom", "ObjectAllValuesFrom", "ObjectMaxCardinality", "ObjectMinCardinality")
    val superClassType = superClass.getClassExpressionType.getName

    superClassType match {
      case m if (quantityModifiers contains m) && !only_taxonomy => {
	val superClass_ = lift2QuantifiedExpression(superClass)
        projectQuantifiedExpression(superClass_, ontology) match {

          case Some((rel, Some(inverseRel), dstClass)) => {
            val dstClasses = splitClass(dstClass)
            val outputEdges = for (dst <- dstClasses)
            yield List(new Triple(ontClass, rel, dst), new Triple(dst, inverseRel, ontClass))
            outputEdges.flatten
          }

          case Some((rel, None, dstClass)) => {
            val dstClasses = splitClass(dstClass)
            for (dst <- dstClasses) yield new Triple(ontClass, rel, dst)
          }
          case None => Nil
        }
      }

      case "Class" => {
	val dst = superClass.asInstanceOf[OWLClass]
        if (bidirectional_taxonomy){
	  new Triple(ontClass, "http://subclassof", dst) :: new Triple(dst, "http://superclassof", ontClass) :: Nil
        }else{
          new Triple(ontClass, "http://subclassof", dst) :: Nil
        }
      }
      case _ => Nil
    }
  }

  def projectQuantifiedExpression(expr:QuantifiedExpression, ontology: OWLOntology): Option[(String, Option[String], OWLClassExpression)] = {

    val rel = expr.getProperty.asInstanceOf[OWLObjectProperty]
    val (relName, inverseRelName) = getRelationInverseNames(rel, ontology)
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

  def processObjectProperty(property: OWLObjectProperty): List[Triple] = {
    Nil
  }

  def getRelationInverseNames(relation: OWLObjectProperty, ontology: OWLOntology): (String, Option[String]) = {
    val relName = relation.getIRI.toString

    if (inverseRelations.contains(relName)){
      (relName, inverseRelations(relName))
    }else{

      val rels = ontology.getInverseObjectPropertyAxioms(relation).asScala.toList
      if (!rels.isEmpty){

        val firstProperty = stripBracket(rels.head.getFirstProperty)
        val secondProperty = stripBracket(rels.head.getSecondProperty) //TODO: remove head and modify the code to deal with lists

        var inverseRelName = secondProperty
        if (secondProperty == relName){
          inverseRelName = firstProperty
        }

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

  val mainLabelURIs = List(
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "rdfs:label",
    "rdfs:comment",
    "http://purl.obolibrary.org/obo/IAO_0000111",
    "http://purl.obolibrary.org/obo/IAO_0000589"
  )
  
  val synonymLabelURIs = List(
    "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
    "http://www.geneontology.org/formats/oboInOWL#hasExactSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
    "http://purl.bioontology.org/ontology/SYN#synonym",
    "http://scai.fraunhofer.de/CSEO#Synonym",
    "http://purl.obolibrary.org/obo/synonym",
    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#FULL_SYN",
    "http://www.ebi.ac.uk/efo/alternative_term",
    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#Synonym",
    "http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#Synonym",
    "http://www.geneontology.org/formats/oboInOwl#hasDefinition",
    "http://bioontology.org/projects/ontologies/birnlex#preferred_label",
    "http://bioontology.org/projects/ontologies/birnlex#synonyms",
    "http://www.w3.org/2004/02/skos/core#altLabel",
    "https://cfpub.epa.gov/ecotox#latinName",
    "https://cfpub.epa.gov/ecotox#commonName",
    "https://www.ncbi.nlm.nih.gov/taxonomy#scientific_name",
    "https://www.ncbi.nlm.nih.gov/taxonomy#synonym",
    "https://www.ncbi.nlm.nih.gov/taxonomy#equivalent_name",
    "https://www.ncbi.nlm.nih.gov/taxonomy#genbank_synonym",
    "https://www.ncbi.nlm.nih.gov/taxonomy#common_name",
    "http://purl.obolibrary.org/obo/IAO_0000118"
  )

  val lexicalAnnotationURIs = mainLabelURIs ::: synonymLabelURIs :::  List(    
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
    "http://purl.org/dc/elements/1.1/description",
    "http://purl.org/dc/terms/description",
    "http://purl.org/dc/elements/1.1/title",
    "http://purl.org/dc/terms/title",    
    "http://purl.obolibrary.org/obo/IAO_0000115",        
    "http://purl.obolibrary.org/obo/IAO_0000600",        
    "http://purl.obolibrary.org/obo/IAO_0000602",
    "http://purl.obolibrary.org/obo/IAO_0000601",
    "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace"
  )

  val excludedAnnotationProperties = List("http://www.geneontology.org/formats/oboInOwl#inSubset", "'http://www.geneontology.org/formats/oboInOwl#id", "http://www.geneontology.org/formats/oboInOwl#hasAlternativeId") //List("rdfs:comment", "http://www.w3.org/2000/01/rdf-schema#comment")

  def annotationAxiom2Edge(annotationAxiom: OWLAnnotationAssertionAxiom): Option[Triple] = {

    val property = annotationAxiom.getProperty.toStringID.toString

    property match {
      case m if true || (lexicalAnnotationURIs contains m) =>  {
        val subject = annotationAxiom.getSubject.toString
        val value = annotationAxiom.getValue
          Some(new Triple(subject, m, stripValue(value.toString)))
      }
      case _ => {
        println("C ",property)
        None
      }
    }
  }

  def stripValue(valueStr: String) = {

    val value = valueStr.replaceAll("\\\\", "")
    value.head match {
      case '"' => value.tail.init
      case '<' => value.tail.init
      case _ => value
    }
  }

  def processAnnotationProperty(annotProperty: OWLAnnotationProperty, ontology: OWLOntology): List[Triple] = {

    val annotations = EntitySearcher.getAnnotations(annotProperty, ontology).asScala.toList
    val property = annotProperty.toStringID.toString

    property match {
      case m if true || (lexicalAnnotationURIs contains m) => {
        val axioms = ontology.getAnnotationAssertionAxioms(annotProperty.getIRI).asScala.toList
        val axx = axioms.map(annotationAxiom2Edge).flatten
        axx
      }

      case _ => {
        println(property)
        Nil
      }
    }
  }

  // Abstract methods
  def project(ontology: OWLOntology, withIndividuals: Boolean, verbose: Boolean): java.util.List[Triple] = Nil.asJava
  def projectAxiom(go_class: OWLClass, axiom: OWLClassAxiom): List[Triple] = Nil
  def projectAxiom(axiom: OWLAxiom): List[org.mowl.Types.Triple] = Nil
  def projectAxiom(axiom: OWLClassAxiom): List[org.mowl.Types.Triple] = Nil
  def projectAxiom(axiom: OWLAxiom, with_individuals: Boolean, verbose: Boolean): List[Triple] = Nil
}

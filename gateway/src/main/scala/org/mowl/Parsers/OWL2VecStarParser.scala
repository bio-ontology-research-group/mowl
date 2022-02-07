package org.mowl.Parsers

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

class OWL2VecStarParser(
  var ontology: OWLOntology,
  var bidirectional_taxonomy: Boolean,
  var only_taxonomy: Boolean,
  var include_literals: Boolean
  // var avoid_properties: java.util.HashSet[String],
  // var additional_preferred_labels_annotations: java.util.HashSet[String],
  // var additional_synonyms_annotations: java.util.HashSet[String],
  // var memory_reasoner: String = "10240"
) extends AbstractParser{


  val inverseRelations = scala.collection.mutable.Map[String, Option[String]]()

//  val dataFactory = OWLManager.createOWLOntologyManager().getOWLDataFactory()

  val searcher = new EntitySearcher()



  override def parse = {

    var edgesFromObjectProperties = List[Triple]()

//    println(ontology.getAxioms(imports).asScala.toList)

    val axioms = ontology.getAxioms(imports).asScala.toList

    var subClassOfAxioms = ListBuffer[OWLSubClassOfAxiom]()
    var equivalenceAxioms = ListBuffer[OWLEquivalentClassesAxiom]()
    var annotationAxioms = ListBuffer[OWLAnnotationAssertionAxiom]()
    var otherAxioms = ListBuffer[OWLAxiom]()

    for (axiom <- axioms){

      axiom.getAxiomType.getName match {
        case "SubClassOf" => subClassOfAxioms += axiom.asInstanceOf[OWLSubClassOfAxiom]
        case "AnnotationAssertion" => {
          if (include_literals)
          annotationAxioms += axiom.asInstanceOf[OWLAnnotationAssertionAxiom]
        }
        case "EquivalentClasses" => equivalenceAxioms += axiom.asInstanceOf[OWLEquivalentClassesAxiom]
          
        
        case _ => {
//          println(axiom)
          otherAxioms += axiom
        }
      }
    }

//    otherAxioms.map(println(_))

    val subClassOfTriples = subClassOfAxioms.flatMap(x => processSubClassAxiom(x.getSubClass.asInstanceOf[OWLClass], x.getSuperClass))
    val equivalenceTriples = equivalenceAxioms.flatMap(
      x => {
        val subClass::superClass::rest= x.getClassExpressionsAsList.asScala.toList

        
        superClass.getClassExpressionType.getName match{
          case "ObjectIntersectionOf" => superClass.asInstanceOf[OWLObjectIntersectionOf].getOperands.asScala.toList.flatMap(processSubClassAxiom(subClass.asInstanceOf[OWLClass], _))

          case _ => Nil
        }

      }
    )
    val annotationTriples = annotationAxioms.map(processAnnotationAxiom(_)).flatten
    
    //val goClasses = ontology.getClassesInSignature().asScala.toList
    //printf("INFO: Number of ontology classes: %d", goClasses.length)
    //val edgesFromClasses = goClasses.foldLeft(List[Triple]()){(acc, x) => acc ::: processOntClass(x)}

    // if (include_literals) {

    //   val objectProperties = ontology.getObjectPropertiesInSignature().asScala.toList//.filter(o => !(avoid_properties contains o))

    //   val edgesFromObjectProperties = objectProperties.foldLeft(List[Triple]()){(acc, x) => acc ::: processObjectProperty(x)}

    //   val annotationProperties = dataFactory.getRDFSLabel ::  ontology.getAnnotationPropertiesInSignature.asScala.toList
    //   val edgesFromAnnotationProperties = annotationProperties.foldLeft(List[Triple]()){(acc, x) => acc ::: processAnnotationProperty(x)}

    //   println("ANNOTATIONS")
    //   println(ontology.getAnnotations.asScala.toList)

    //   (edgesFromClasses ::: edgesFromObjectProperties  ::: edgesFromAnnotationProperties).asJava

    // }else {
    //   (edgesFromClasses ::: edgesFromObjectProperties).asJava

    // }

    //println(ontology.getRBoxAxioms(imports))

    (subClassOfTriples.toList ::: equivalenceTriples.toList ::: annotationTriples.toList).asJava
  }

   






  // CLASSES PROCESSING

  override def processOntClass(ontClass: OWLClass): List[Triple] = {

    var annotationEdges = List[Triple]()

    if (include_literals){ //ANNOTATION PROCESSSING
      val annotProperties = ontClass.getAnnotationPropertiesInSignature.asScala.toList

      val annotationAxioms = ontology.getAnnotationAssertionAxioms(ontClass.getIRI).asScala.toList
      annotationEdges = annotationAxioms.map(annotationAxiom2Edge).flatten
    }

    val axioms = ontology.getAxioms(ontClass, imports).asScala.toList
    val edges = axioms.flatMap(parseAxiom(ontClass, _: OWLClassAxiom))
    edges ::: annotationEdges
  }



  def parseAxiom(goClass: OWLClass, axiom: OWLClassAxiom): List[Triple] = {

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

  def processSubClassAxiom(subClass: OWLClass, superClass: OWLClassExpression): List[Triple] = {

    val quantityModifiers = List("ObjectSomeValuesFrom", "ObjectAllValuesFrom", "ObjectMaxCardinality", "ObjectMinCardinality")

    val superClassType = superClass.getClassExpressionType.getName

    superClassType match {

      case m if (quantityModifiers contains m) && !only_taxonomy => {

	val superClass_ = lift2QuantifiedExpression(superClass)

        parseQuantifiedExpression(superClass_) match {

          case Some((rel, Some(inverseRel), dstClass)) => {
            val dstClasses = splitClass(dstClass)

            val outputEdges = for (dst <- dstClasses)
            yield List(new Triple(subClass, rel, dst), new Triple(dst, inverseRel, subClass))

            outputEdges.flatten
          }

          case Some((rel, None, dstClass)) => {
            val dstClasses = splitClass(dstClass)

            for (dst <- dstClasses) yield new Triple(subClass, rel, dst)

          }

          case None => Nil
        }
      }

      case "Class" => {
	val dst = superClass.asInstanceOf[OWLClass]
        if (bidirectional_taxonomy){
	  new Triple(subClass, "subClassOf", dst) :: new Triple(dst, "superClassOf", subClass) :: Nil
        }else{
          new Triple(subClass, "subClassOf", dst) :: Nil
        }

      }
      case _ => Nil

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

  def parseSubClassOrEquivAxiom(goClass: OWLClass, superClass: OWLClassExpression): List[Triple] = {

     val quantityModifiers = List("ObjectSomeValuesFrom", "ObjectAllValuesFrom", "ObjectMaxCardinality", "ObjectMinCardinality")

     val superClassType = superClass.getClassExpressionType.getName

    superClassType match {

      case m if (quantityModifiers contains m) && !only_taxonomy => {

	val superClass_ = lift2QuantifiedExpression(superClass)

        parseQuantifiedExpression(superClass_) match {

          case Some((rel, Some(inverseRel), dstClass)) => {
            val dstClasses = splitClass(dstClass)

            val outputEdges = for (dst <- dstClasses)
            yield List(new Triple(goClass, rel, dst), new Triple(dst, inverseRel, goClass))

            outputEdges.flatten
          }

          case Some((rel, None, dstClass)) => {
            val dstClasses = splitClass(dstClass)

            for (dst <- dstClasses) yield new Triple(goClass, rel, dst)

          }

          case None => Nil
        }
      }

      case "Class" => {
	val dst = superClass.asInstanceOf[OWLClass]
        if (bidirectional_taxonomy){
	  new Triple(goClass, "subClassOf", dst) :: new Triple(dst, "superClassOf", goClass) :: Nil
        }else{
          new Triple(goClass, "subClassOf", dst) :: Nil
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

  def processObjectProperty(property: OWLObjectProperty): List[Triple] = {


    Nil

  }

  def getRelationInverseNames(relation: OWLObjectProperty): (String, Option[String]) = {
    val relName = relation.getIRI.toString

    if (inverseRelations.contains(relName)){
      (relName, inverseRelations(relName))
    }else{

      val rels = ontology.getInverseObjectPropertyAxioms(relation).asScala.toList

      if (!rels.isEmpty){
       // println(relation, rels.head.getFirstProperty, rels.head.getSecondProperty)

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
//        println("N ", property)
        
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

  def processAnnotationProperty(annotProperty: OWLAnnotationProperty): List[Triple] = {

//    println("Annotation Property: ")
//    println(ontology.getAxioms(annotProperty))

    val annotations = EntitySearcher.getAnnotations(annotProperty, ontology).asScala.toList

    println(annotations)

    println(dataFactory.getRDFSLabel)
  //    val deepAnnotations = annotations.flatMap(a => a.getAnnotations.asScala.toList)

    val property = annotProperty.toStringID.toString

    property match {
      case m if true || (lexicalAnnotationURIs contains m) => {
        val axioms = ontology.getAnnotationAssertionAxioms(annotProperty.getIRI).asScala.toList
        println(property, " ", axioms.length)
//        println(axioms)
        val axx = axioms.map(annotationAxiom2Edge).flatten
//        println(property, " ", axx.length)
        axx
//        Nil
      }

      case _ => {
        println(property)
        Nil
      }
    }
//    println(ontology.getAnnotationAssertionAxioms(annotProperty.getIRI))

  //  Nil
  }

}

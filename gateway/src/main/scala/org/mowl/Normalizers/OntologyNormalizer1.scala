package org.mowl.Normalizers

// OWL API imports
import org.semanticweb.owlapi.model._
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.parameters.Imports
import uk.ac.manchester.cs.owl.owlapi.OWLObjectSomeValuesFromImpl


import org.mowl.Utils.removeBrackets

// Java imports
import java.io.File


import collection.JavaConverters._
import scala.collection.mutable.Map

import org.mowl.Types._


class OntologyNormalizer1() {

  private var _ontology: OWLOntology = _
  private val ont_manager = OWLManager.createOWLOntologyManager()
  println(s"INFO: Start loading ontology")
  private val data_factory = ont_manager.getOWLDataFactory()
  println("INFO: Finished creating data factory")


  def ontology = _ontology

  def normalize(ontology: OWLOntology)= {
    _ontology = ontology

    val imports = Imports.fromBoolean(false)
    val go_classes = ontology.getClassesInSignature(imports).asScala.toList
    println(s"INFO: Number of ontology classes: ${go_classes.length}")

    val axioms = go_classes.foldLeft(List[GCI]()){(acc, x) => acc :::  processOntClass(x)}
    val grouped_axioms = axioms.groupBy(_.getClassName)
    val finalMap = grouped_axioms.map(e => (e._1, e._2.asJava))
    
    finalMap.asJava
  }


  def mergeMap[K, V](a: Map[K, Iterable[V]], b: Map[K, Iterable[V]]): Map[K, Iterable[V]] = {
    a ++ b.map { case (k, v) => k -> (v ++ a.getOrElse(k, Iterable.empty)) }
  }


  def processOntClass(go_class: OWLClass): List[GCI] = {
    val axioms = ontology.getAxioms(go_class).asScala.toList
    axioms.flatMap(parseAxiom(go_class, _: OWLClassAxiom))
  }

  def parseAxiom(go_class: OWLClass, axiom: OWLClassAxiom) : List[GCI] = {

    val axiomType = axiom.getAxiomType().getName()
    axiomType match {
      case "EquivalentClasses" => {
        var ax = axiom.asInstanceOf[OWLEquivalentClassesAxiom].getClassExpressionsAsList.asScala.toList
        ax.filter(_ != go_class).flatMap(parseEquivClassAxiom(go_class, _: OWLClassExpression))
      }
      case "SubClassOf" => {
        var ax = axiom.asInstanceOf[OWLSubClassOfAxiom]
        parseSubClassAxiom(ax.getSubClass.asInstanceOf[OWLClass], ax.getSuperClass)
      }
      // case "DisjointClasses" => {
      //   var ax = axiom.asInstanceOf[OWLDisjointClassesAxiom].getClassExpressionsAsList.asScala.toList
      //   ax.filter(_ != go_class).flatMap(parseDisjointnessAxiom(go_class, _: OWLClassExpression))
      // }
      case _ => {
        println(s"Not parsing axiom $axiomType")
        Nil
//        throw new Exception()
      }
    }
  }
//}

  def initializeMap(): Map[String, List[GCI]] = {
    val initialMap = Map[String, List[GCI]]()
    initialMap("type1") = List[GCIType1]()
    initialMap("type2") = List[GCIType2]()
    initialMap("type3") = List[GCIType3]()
    initialMap("type4") = List[GCIType4]()
    initialMap("type5") = List[GCIType5]()

    initialMap
  }

  /////////////////////////////////////////////
  def parseEquivClassAxiom(go_class: OWLClass, rightSideExpr: OWLClassExpression): List[GCI] =  {


    val rightSideType = rightSideExpr.getClassExpressionType.getName

    rightSideType match {

      case "Class" => { // A equiv B
        val rightOWLClass = rightSideExpr.asInstanceOf[OWLClass]
        val axiom1 = new GCIType1(go_class, rightOWLClass)
        val axiom2 = new GCIType1(rightOWLClass, go_class)
        axiom1 :: axiom2 :: Nil
      }
      case _ => {

        val axiomsOneDirection = parseSubClassAxiom(go_class, rightSideExpr)
        val axiomsOtherDirecion = parseSubClassAxiomComplex(rightSideExpr, go_class)
        
        axiomsOneDirection ::: axiomsOtherDirecion

      }
    }
  }

  def parseDisjointnessAxiom() = {}
  // def parseDisjointnessAxiom(go_class: OWLClass, rightSideExpr: OWLClassExpression) = {

  //   val rightSideType = rightSideExpr.getClassExpressionType.getName

  //   rightSideType match {

  //     case "Class" => {
  //       val intersectionObject = "intersection_" + go_class.toStringID + "_" + rightSideExpr.asInstanceOf[OWLClass].toStringID

  //       val entailment = subclassMorphism(intersectionObject, Bottom )
  //       val intersection_edges = parseIntersection(go_class, rightSideExpr, intersectionObject)
  //       entailment :: intersectionEdges
  //     }

  //     case _ => {
  //       val intersectionObject = "intersection_" + go_class.toStringID + "_others"
  //       val entailment = entailmentMorphism(intersectionObject, Bottom)
  //       val intersectionEdges = parseIntersection(go_class, rightSideExpr, intersectionObject)

  //       entailment :: intersectionEdges

  //     }

  //   }


 //  }

  def parseSubClassAxiom(go_class: OWLClass, superClass: OWLClassExpression): List[GCI] = {

    val rightSideType = superClass.getClassExpressionType.getName

    rightSideType match {
      case "Class" => {
        val rightOWLClass = superClass.asInstanceOf[OWLClass]
        val axiom1 =new GCIType1(go_class, rightOWLClass)
        axiom1 :: Nil

      }

      case "ObjectIntersectionOf" => {
        val rightSideOperands = superClass.asInstanceOf[OWLObjectIntersectionOf].getOperands.asScala.toList
        rightSideOperands.flatMap(parseSubClassAxiom(go_class, _))

      }

      case "ObjectSomeValuesFrom" => {
        val rightSideEx = superClass.asInstanceOf[OWLObjectSomeValuesFrom]
        val property = rightSideEx.getProperty.asInstanceOf[OWLObjectProperty]
        val filler = rightSideEx.getFiller

        val fillerType = filler.getClassExpressionType.getName

        fillerType match {

          case "Class" => {
           new GCIType2(go_class, property, filler.asInstanceOf[OWLClass]) :: Nil
          }
          case _ => {
            println(s"Subclass axiom: existential filler in right side not suported. Type is $fillerType")
            Nil
          }

        }

      }

      case _ => {
        println(s"Subclass axiom: rightside not supported. Type is $rightSideType")
        Nil

      }
    }


  }



    def parseSubClassAxiomComplex(subClass: OWLClassExpression, go_class: OWLClass): List[GCI] = {

    val leftSideType = subClass.getClassExpressionType.getName

    leftSideType match {
      case "Class" => {
        val leftOWLClass = subClass.asInstanceOf[OWLClass]
        val axiom1 = new GCIType1(leftOWLClass, go_class)
        axiom1 :: Nil

      }

      case "ObjectIntersectionOf" => {
        val leftSideOperands = subClass.asInstanceOf[OWLObjectIntersectionOf].getOperands.asScala.toList
        val leftSideTypes = leftSideOperands.map(_.getClassExpressionType.getName)

        leftSideOperands.length match {
          case 1 => parseSubClassAxiomComplex(leftSideOperands(0), go_class)
          case 2 => {
            if (leftSideTypes.forall(_ == "Class")){
              new GCIType5(toOWLClass(leftSideOperands(0)), toOWLClass(leftSideOperands(1)), go_class) :: Nil
            }else if(leftSideTypes(0) == "Class" && leftSideTypes(1) == "ObjectSomeValuesFrom"){
              val left_subclass = leftSideOperands(0).asInstanceOf[OWLClass]
              val existential_part = leftSideOperands(1).asInstanceOf[OWLObjectSomeValuesFrom]
              val property = existential_part.getProperty.asInstanceOf[OWLObjectProperty]
              val filler = existential_part.getFiller

              val fillerType = filler.getClassExpressionType.getName
              fillerType match {
                case "Class" => new GCIType6(left_subclass, property, filler.asInstanceOf[OWLClass], go_class) :: Nil
                case _ => {
                  println(s"Subclass complex axiom: left existential filler too complex $fillerType")
                  Nil
                }
              }
            }
            else{
              println(s"Subclass complex axiom: left side expression not supported $leftSideTypes")
              Nil
            }
          }
          case n => {
            println(s"Intersection too large: $n")
            Nil
          }
        }
      }

      case "ObjectSomeValuesFrom" => {
        val leftSideEx = subClass.asInstanceOf[OWLObjectSomeValuesFrom]
        val property = leftSideEx.getProperty.asInstanceOf[OWLObjectProperty]
        val filler = leftSideEx.getFiller

        val fillerType = filler.getClassExpressionType.getName

        fillerType match {

          case "Class" => {
           new GCIType3(property, filler.asInstanceOf[OWLClass], go_class) :: Nil
          }
          case _ => {
            println(s"Subclass complex axiom: existential filler in left side not suported. Type is $fillerType")
            Nil
          }

        }

      }

      case _ => {
        println(s"Subclass complex axiom: left side not supported. Type is $leftSideType")
        Nil

      }
    }


  }


  

  ///////////////////////////////////////

  def getClassString(owlClass: OWLClass) = {
    removeBrackets(owlClass.toStringID)
  }

  def getPropertyString(owlProperty: OWLObjectProperty) = {
    removeBrackets(owlProperty.toString)
  }
  sealed trait GCI {
    def getClassName() : String
  }


  // a SubClassOf b
  case class GCIType1(val subclass: String, val superclass: String) extends GCI {
    def this(subclass: OWLClass, superclass: OWLClass) = this(getClassString(subclass), getClassString(superclass))

    def getClassName() = "gci_type_1"
  }

  // a SubClassOf r some b
  case class GCIType2(val subclass: String, val obj_property: String, val filler: String) extends GCI {
    def this(subclass: OWLClass, obj_property: OWLObjectProperty, filler: OWLClass) = this(getClassString(subclass), getPropertyString(obj_property), getClassString(filler))
    def getClassName() = "gci_type_2"
  }

  // r some b SubClassOf a
  case class GCIType3(val obj_property: String, val filler: String, val superclass: String) extends GCI {
    def this(obj_property: OWLObjectProperty, filler: OWLClass, superclass: OWLClass) = this(getPropertyString(obj_property), getClassString(filler), getClassString(superclass))
    def getClassName() = "gci_type_3"
  }

  // a and b SubClassOf c
  case class GCIType4(val left_subclass: String, val right_subclass: String, val superclass: String) extends GCI {
    def this(left_subclass: OWLClass, right_subclass: OWLClass, superclass: OWLClass) = this(getClassString(left_subclass), getClassString(right_subclass),  getClassString(superclass))
    def getClassName() = "gci_type_4"
  }

  // a SubClassOf b and c
  case class GCIType5(val subclass: String, val left_superclass: String, val right_superclass: String) extends GCI {
    def this(subclass: OWLClass, left_superclass: OWLClass, right_superclass: OWLClass) = this(getClassString(subclass), getClassString(left_superclass), getClassString(right_superclass))
    def getClassName() = "gci_type_5"
  }

  // a and r some c SubClassOf d
  case class GCIType6(val left_subclass: String, val obj_property: String, val filler: String, val superclass: String) extends GCI {
    def this(left_subclass: OWLClass, obj_property: OWLObjectProperty, filler: OWLClass, superclass: OWLClass) = this(getClassString(left_subclass), getPropertyString(obj_property), getClassString(filler), getClassString(superclass))
    def getClassName() = "gci_type_6"
  }
  

  def toOWLClass(expr: OWLClassExpression) : OWLClass = {
    expr.asInstanceOf[OWLClass]
  }

}

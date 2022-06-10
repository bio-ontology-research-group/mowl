package org.mowl.Projectors

// OWL API imports
import org.semanticweb.owlapi.model._
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.parameters.Imports
import uk.ac.manchester.cs.owl.owlapi.OWLObjectSomeValuesFromImpl


// Java imports
import java.io.File


import collection.JavaConverters._

import org.mowl.Types._


class CatProjector(var ontology: OWLOntology, var bidirectional: Boolean) {

  private val ont_manager = OWLManager.createOWLOntologyManager()
  println(s"INFO: Start loading ontology")
  private val data_factory = ont_manager.getOWLDataFactory()
  println("INFO: Finished creating data factory")
    
  var rel_counter = 0

  def project = {
           
    val axioms = ontology.getAxioms()
    val imports = Imports.fromBoolean(false)

    val go_classes = ontology.getClassesInSignature(imports).asScala.toList


    println(s"INFO: Number of GO classes: ${go_classes.length}")


    val edges = go_classes.foldLeft(List[Triple]()){(acc, x) => acc ::: processGOClass(x)}

    val nodes = getNodes(edges)

    // val id_edges = nodes.map((x) => new Triple(x, "http://identity", x)).toList

    edges.asJava
    //        (id_edges ::: edges).asJava
  }





  def processGOClass(go_class: OWLClass): List[Triple] = {
    val axioms = ontology.getAxioms(go_class).asScala.toList

    val edges = axioms.flatMap(parseAxiom(go_class, _: OWLClassAxiom))
    edges
  }

  def parseAxiom(go_class: OWLClass, axiom: OWLClassAxiom): List[Triple] = {
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


  /////////////////////////////////////////////
  def parseEquivClassAxiom(go_class: OWLClass, rightSideExpr: OWLClassExpression, prevRel: Option[String] = None, origin: String = "Not specified"): List[Triple] =  {


    val rightSideType = rightSideExpr.getClassExpressionType.getName

    rightSideType match {

      case "Class" => {
        val rightOWLClass = rightSideExpr.asInstanceOf[OWLClass]
        val subclassEntailment1 = subclassMorphism(go_class, rightOWLClass)
        val subclassEntailment2 = subclassMorphism(rightOWLClass, go_class)

        subclassEntailment1 :: subclassEntailment2 :: Nil
      }
      case _ => {

        val rightSideObj = "equivalentTo_" + go_class.toStringID
        val entailment1 = subclassMorphism(rightSideObj, go_class)
        val otherEntailments = parseSubClassAxiom(go_class, rightSideExpr, rightSideObj)
        
        entailment1 :: otherEntailments

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

  def parseSubClassAxiom(go_class: OWLClass, superClass: OWLClassExpression, nameRightSide: String  = ""): List[Triple] = {

    val rightSideType = superClass.getClassExpressionType.getName

    rightSideType match {
      case "Class" => {
        val subclassTriple = subclassMorphism(go_class, superClass.asInstanceOf[OWLClass])
        val superclassTriple = superclassMorphism(superClass.asInstanceOf[OWLClass], go_class)
        subclassTriple :: superclassTriple :: Nil
      }
      case _ => {
        val superClassObject = nameRightSide match {
          case "" => "superclassOf_" + go_class.toStringID
          case m => m
        }
        
        val entailment = subclassMorphism(go_class, superClassObject)
        val reverseEntailment = superclassMorphism(superClassObject, go_class)
        val rightEdges = parseExpression(superClass, superClassObject)

        entailment :: reverseEntailment :: rightEdges

      }
    }


  }


  /////////////////////////////////////////////

  def parseExpression(expression: OWLClassExpression, origin: String) : List[Triple] = {
    val expressionType = expression.getClassExpressionType.getName

    expressionType match {
      case "ObjectIntersectionOf" => {
        val operands = expression.asInstanceOf[OWLObjectIntersectionOf].getOperands.asScala.toList
        val edges = operands.flatMap((x) => generateProjectionMorphism(origin, x))
        edges
      }

      case "ObjectSomeValuesFrom" => {
        val expr = expression.asInstanceOf[OWLObjectSomeValuesFrom]
        val filler = expr.getFiller
        val property = expr.getProperty

        filler.getClassExpressionType.getName match {
          case "Class" => {
            new Triple(origin, property.toString, filler.asInstanceOf[OWLClass]) :: Nil
            
          }
        }
        
      }

      case _ =>  {
        println(s"Not parsing axiom $expressionType")
        Nil
      }
    }
  }

  

  def generateProjectionMorphism(big: String, projected: OWLClassExpression) : List[Triple] = {
    val exprType = projected.getClassExpressionType.getName

        exprType match {
          case "Class" => projectionMorphism(big, projected) :: Nil

          // case "ObjectComplementOf" => {
          //       val operand = projected.asInstanceOf[OWLObjectComplementOf].getOperand
          //       val operandType = operand.getClassExpressionType.getName

          //       operandType match {
          //         case "Class" => {
          //           val negative = "not_" + operand.toStringID
          //           projectionMorphism(go_class, negative)
          //           }
          //           case _ => {
          //             Nil
          //           }
          //       }
          //   }

          //   case "ObjectSomeValuesFrom" => {
          //       val projType = projected.asInstanceOf[OWLObjectSomeValuesFrom]
                
          //       val(rel, dst_class) = parseQuantifiedExpression(Existential(proj_class)) 


          //       prevRel match {
          //           case None => parseIntersection(go_class, dst_class, Some(rel), "rec intersection OSV") // simple case

          //           case Some(r) => parseIntersection(go_class, dst_class, Some(r + "_" + rel))

          //           case _ => throw new Exception(s"Complex structure in ObjectSomeValuesFrom $origin\n$go_class\n$projected_expr")
          //       }
          //   }


          //   case "ObjectAllValuesFrom" => {
          //       val proj_class = projected_expr.asInstanceOf[OWLObjectAllValuesFrom]
                
          //       val(rel, dst_class) = parseQuantifiedExpression(Universal(proj_class)) 

          //       prevRel match {
          //           case None => parseIntersection(go_class, dst_class, Some(rel), "rec intersection OAV") // simple case

          //           // case Some(r) => parseIntersection(go_class, dst_class, Some(r + "_" + rel))

          //           case _ => throw new Exception(s"Complex structure in ObjectSomeValuesFrom $origin\n$go_class\n$projected_expr")
          //       }
          //   }


          //   case "ObjectIntersectionOf" => {
          //       val proj_class = projected_expr.asInstanceOf[OWLObjectIntersectionOf]
          //       val exactCardinality = checkExactCardinality(proj_class)
                
          //       exactCardinality match {
          //           case None => {
          //               val proj_class = projected_expr.asInstanceOf[OWLObjectIntersectionOf].getOperands.asScala.toList

                
          //              proj_class.flatMap(parseIntersection(go_class, _: OWLClassExpression, prevRel, "nested intersection"))
          //           }
          //           case Some(expr) => parseIntersection(go_class, expr, prevRel, "exactCardinality")
          //       }
          //   }

          //   case "ObjectMinCardinality" => {
          //       val proj_class = projected_expr.asInstanceOf[OWLObjectMinCardinality]
                
          //       val(rel, src_class) = parseQuantifiedExpression(MinCardinality(proj_class), true) 

          //       val src_type = src_class.getClassExpressionType.getName
                
          //       prevRel match {
          //           case None => parseIntersection(go_class, src_class, Some(rel), "rec union OMC") // simple case

          //           case _ => throw new Exception(s"Complex structure in ObjectMinCardinality $src_type")

          //       }
          //   }

          //   case "ObjectUnionOf" => {
          //       val proj_class = projected_expr.asInstanceOf[OWLObjectUnionOf]
                
          //       //throw new Exception(s"Not parsed complex intersection in union: $origin\n$go_class")
          //       println(s"PARSING WARNING: Not parsed complex nested union in intersection: $origin\n$go_class")
          //       List()
        
          //   }
          case "ObjectSomeValuesFrom" => {
            val projected_ = projected.asInstanceOf[OWLObjectSomeValuesFrom]
            val filler = projected_.getFiller
            val objProp = projected_.getProperty

            filler.getClassExpressionType.getName match {
              case "Class" => {
                val filler_ = filler.asInstanceOf[OWLClass]
                val existential_object = "exists_" + objProp.toString + "_" + filler_.toStringID
                val projMorphism = projectionMorphism(big, existential_object)
                val otherMorphism = new Triple(existential_object, objProp.toString, filler_)

                projMorphism :: otherMorphism :: Nil
              }
              case _ => {
                println("Not parsing complex filler in projection morphism: $filler")
                Nil
              }
            }

          }
          case _ =>  {
            println(s"Not parsing projection from $big into $projected")
            Nil
            
          }
        }
  }



    def parseUnion(go_class: OWLClass, injected_expr: OWLClassExpression, prevRel: Option[String] = None, origin: String = "Union"): List[Triple] = {
        val exprType = injected_expr.getClassExpressionType.getName

        exprType match {
            case "Class" => injectionMorphism(injected_expr, go_class, prevRel) :: Nil

            case "ObjectComplementOf" => {
                val operand = injected_expr.asInstanceOf[OWLObjectComplementOf].getOperand
                val operandType = operand.getClassExpressionType.getName

                operandType match {
                    case "Class" => {
                        val neg = negationMorphism(operand)
                        val injection = parseUnion(go_class, operand, prevRel,"rec union OC")
                        neg :: injection
                    }
                    case _ => {
                        val injected_NNF = injected_expr.getNNF
                        parseUnion(go_class, injected_NNF, prevRel)
                    }
                }
            }

            case "ObjectSomeValuesFrom" => {
                val inj_class = injected_expr.asInstanceOf[OWLObjectSomeValuesFrom]
                
                val(rel, src_class) = parseQuantifiedExpression(Existential(inj_class), true) 

                val src_type = src_class.getClassExpressionType.getName

                prevRel match {
                    case None => parseUnion(go_class, src_class, Some(rel), "rec union OSV") // simple case

                    case Some(r) => parseUnion(go_class, src_class, Some(r + "_" + rel))

                    case _ => throw new Exception(s"Complex structure in ObjectSomeValuesFrom $src_type\n$origin\n$go_class\n$injected_expr")

                }
            }
         
            case "ObjectAllValuesFrom" => {
                val inj_class = injected_expr.asInstanceOf[OWLObjectAllValuesFrom]
                
                val(rel, src_class) = parseQuantifiedExpression(Universal(inj_class), true) 

                val src_type = src_class.getClassExpressionType.getName
                
                prevRel match {
                    case None => parseUnion(go_class, src_class, Some(rel), "rec union OAV") // simple case

                    case _ => throw new Exception(s"Complex structure in ObjectAllValuesFrom $src_type")

                }
            }

            case "ObjectMinCardinality" => {
                val inj_class = injected_expr.asInstanceOf[OWLObjectMinCardinality]
                
                val(rel, src_class) = parseQuantifiedExpression(MinCardinality(inj_class), true) 

                val src_type = src_class.getClassExpressionType.getName
                
                prevRel match {
                    case None => parseUnion(go_class, src_class, Some(rel), "rec union OMC") // simple case

                    case _ => throw new Exception(s"Complex structure in ObjectMinCardinality $src_type")

                }
            }

            case "ObjectExactCardinality" => {
                val inj_class = injected_expr.asInstanceOf[OWLObjectExactCardinality].asIntersectionOfMinMax
                parseUnion(go_class, inj_class, prevRel, "Object exact cardinality")
            }

            case "ObjectIntersectionOf" => {
                val inj_class = injected_expr.asInstanceOf[OWLObjectIntersectionOf]
                val exactCardinality = checkExactCardinality(inj_class)
                
                exactCardinality match {
                    case None => {
                        //throw new Exception(s"Not parsed complex intersection in union: $origin\n$go_class")
                        println(s"PARSING WARNING: Not parsed nested intersection in union: $origin\n$go_class")
                        List()
                    }
                    case Some(expr) => parseUnion(go_class, expr, prevRel, "exactCardinality")
                }
            }

            case _ =>  throw new Exception(s"Not parsing Union ($origin) operand $exprType\n$go_class")
        }

    }

    def parseQuantifiedExpression(expr: QuantifiedExpression, inverse: Boolean = false) = {
        
        var relation = expr.getProperty.asInstanceOf[OWLObjectProperty]

        val rel = getRelationName(relation, inverse)

        val dst_class = expr.getFiller

        if (inverse){
            var inv_relation = relation.getInverseProperty
            if (!inv_relation.isAnonymous){
                (rel, dst_class)
            }else{
                ("inv_" + rel, dst_class)
            }
        }else{
            (rel, dst_class)
        }
        
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

    def checkExactCardinality(expr: OWLObjectIntersectionOf) = {
        val operands = expr.getOperands.asScala.toList

        lazy val length_2 = operands.length == 2
        lazy val fst :: snd :: xs = operands

        lazy val fst_type = fst.getClassExpressionType.getName
        lazy val snd_type = snd.getClassExpressionType.getName

        

        if (length_2) {
            if (fst_type == "ObjectMinCardinality" && snd_type == "ObjectMaxCardinality") {
                val fst_card = fst.asInstanceOf[OWLObjectMinCardinality].getCardinality
                val snd_card = snd.asInstanceOf[OWLObjectMaxCardinality].getCardinality
                
                (fst_card == snd_card) match {
                    case true => Some(fst)
                    case false => None
                }
            }else if (fst_type == "ObjectMaxCardinality" && snd_type == "ObjectMinCardinality") {
                val fst_card = fst.asInstanceOf[OWLObjectMaxCardinality].getCardinality
                val snd_card = snd.asInstanceOf[OWLObjectMinCardinality].getCardinality
                
                (fst_card == snd_card) match {
                    case true => Some(snd)
                    case false => None
                }
            }else{
                None
            }

        }else{
            None
        }

    }

    ///////////////////////////////////////

    def negationMorphism(go_class: OWLClassExpression) = {
        val go_class_OWLClass = go_class.asInstanceOf[OWLClass]
        new Triple(s"Not_${goClassToStr(go_class_OWLClass)}", "negate", go_class_OWLClass)
    }

    def injectionMorphism(src: OWLClassExpression, dst: OWLClassExpression, rel: Option[String] = None) = {
        val src_OWLClass = src.asInstanceOf[OWLClass]
        val dst_OWLClass = dst.asInstanceOf[OWLClass]
        rel match {
            case Some(r) => new Triple(src_OWLClass, "injects_" + r, dst_OWLClass)
            case None => new Triple(src_OWLClass, "injects", dst_OWLClass)
        }
    }

    def projectionMorphism(src: OWLClassExpression, dst: OWLClassExpression) = {
    val src_OWLClass = src.asInstanceOf[OWLClass]
    val dst_OWLClass = dst.asInstanceOf[OWLClass]
    new Triple(src_OWLClass, "projects", dst_OWLClass)
    
    }
    def projectionMorphism(src: String, dst: OWLClassExpression) = {
      val src_OWLClass = src
      val dst_OWLClass = dst.asInstanceOf[OWLClass]
      new Triple(src_OWLClass, "projects", dst_OWLClass)
    
    }
    def projectionMorphism(src: OWLClassExpression, dst: String) = {
    val src_OWLClass = src.asInstanceOf[OWLClass]
    val dst_OWLClass = dst
    new Triple(src_OWLClass, "projects", dst_OWLClass)
    
    }
    def projectionMorphism(src: String, dst: String) = {
      val src_OWLClass = src
    val dst_OWLClass = dst
    new Triple(src_OWLClass, "projects", dst_OWLClass)
    
    }

    def subclassMorphism(src: OWLClassExpression, dst: OWLClassExpression) = {
      val src_OWLClass = src.asInstanceOf[OWLClass]
      val dst_OWLClass = dst.asInstanceOf[OWLClass]
      new Triple(src_OWLClass, "subclass", dst_OWLClass)
      
  }
  def subclassMorphism(src: OWLClassExpression, dst: String) = {
      val src_OWLClass = src.asInstanceOf[OWLClass]
      val dst_OWLClass = dst
      new Triple(src_OWLClass, "subclass", dst_OWLClass)
      
  }
  def subclassMorphism(src: String, dst: OWLClassExpression) = {
    val src_OWLClass = src
    val dst_OWLClass = dst.asInstanceOf[OWLClass]
    new Triple(src_OWLClass, "subclass", dst_OWLClass)
      
  }


  def superclassMorphism(src: OWLClassExpression, dst: OWLClassExpression) = {
      val src_OWLClass = src.asInstanceOf[OWLClass]
      val dst_OWLClass = dst.asInstanceOf[OWLClass]
      new Triple(src_OWLClass, "superclass", dst_OWLClass)
      
  }
  def superclassMorphism(src: OWLClassExpression, dst: String) = {
      val src_OWLClass = src.asInstanceOf[OWLClass]
      val dst_OWLClass = dst
      new Triple(src_OWLClass, "superclass", dst_OWLClass)
      
  }
  def superclassMorphism(src: String, dst: OWLClassExpression) = {
    val src_OWLClass = src
    val dst_OWLClass = dst.asInstanceOf[OWLClass]
    new Triple(src_OWLClass, "superclass", dst_OWLClass)
      
  }

}

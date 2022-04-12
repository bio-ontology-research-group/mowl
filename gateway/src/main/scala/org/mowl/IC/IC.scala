package org.mowl.IC

import slib.sml.sm.core.utils._
import slib.sml.sm.core.engine._
import slib.graph.model.impl.repo._
import slib.graph.io.conf._
import slib.graph.io.util._
import slib.graph.io.loader._
import slib.graph.algo.utils._
import slib.sml.sm.core.metrics.ic.utils._
import slib.graph.model.impl.graph.memory._
import org.openrdf.model.vocabulary._
import slib.graph.model.impl.graph.elements.Edge

import java.io.File
import java.io.FileOutputStream
import java.util.HashMap
import java.util.ArrayList

import collection.JavaConverters._

import org.semanticweb.owlapi.model._



object IC {


  def computeIC(ontology:OWLOntology, annotationsJ:HashMap[String, ArrayList[String]]) = {

    val annotations = annotationsJ.asScala.mapValues(_.asScala)
    val tmpFile = File.createTempFile("tmp", ".owl")
    val filePath = tmpFile.getAbsolutePath
    val outputStream = new FileOutputStream(tmpFile)
    ontology.saveOntology(outputStream)

    val factory = URIFactoryMemory.getSingleton
    val graphURI = factory.getURI("http://graph/")
    
    factory.loadNamespacePrefix("GRAPH", graphURI.toString())
    val graph = new GraphMemory(graphURI)

    val dataConf = new GDataConf(GFormat.RDF_XML, filePath)

    GraphLoaderGeneric.populate(dataConf, graph)

    // val virtualRoot = factory.getURI("http://graph/virtualRoot")
    // graph.addV(virtualRoot)

    // val rooting = new GAction(GActionType.REROOTING)
    // rooting.addParameter("root_uri", virtualRoot.stringValue())
    // GraphActionExecutor.applyAction(factory, rooting, graph)

    //Add anotations
    for ((external_entity, ontClasses) <- annotations) {
      val entityURI = factory.getURI("http://" + external_entity)

      for (ontClass <- ontClasses) {
        val uri = factory.getURI("http://graph/" + ontClass)
        val e = new Edge(entityURI, RDF.TYPE, uri)
        graph.addE(e)
      }

    }

    val engine = new SM_Engine(graph)
    val icConf = new IC_Conf_Corpus("ResnikIC", SMConstants.FLAG_IC_ANNOT_RESNIK_1995_NORMALIZED)

    val ics = engine.computeIC(icConf)
    ics
  }


}




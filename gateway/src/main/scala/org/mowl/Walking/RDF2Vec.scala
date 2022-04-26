package org.mowl.Walking

import collection.JavaConverters._
import java.io._
import java.util.{HashMap, ArrayList}
import scala.collection.mutable.{MutableList, ListBuffer, Map, ArrayBuffer}
import util.control.Breaks._
import java.util.concurrent.{ExecutorService, Executors}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.concurrent.{ Await, Future }
import scala.concurrent.{ExecutionContext, ExecutionContextExecutorService}
import scala.util.{Failure, Success, Try}
import org.mowl.Edge

class RDF2Vec (
  var edges: ArrayList[Edge],
  var numWalks: Int,
  var walkLength: Int,
  var workers: Int,
  var outfile: String) {


  val edgesSc = edges.asScala.map(x => (x.src, x.rel, x.dst))
  val entities = edgesSc.map(x => List(x._1, x._2, x._3)).flatten.toSet
  val mapEntsIdx = entities.zip(Range(0, entities.size, 1)).toMap
  val mapIdxEnts = Range(0, entities.size, 1).zip(entities).toMap
  val entsIdx = entities.map(mapEntsIdx(_))
  val nodes = edgesSc.map(x => List(x._1, x._3)).flatten.toSet
  val nodesIdx = nodes.map(mapEntsIdx(_))
  val graph = processEdges()

  val rand = scala.util.Random
  val (pathsPerWorker, newWorkers) = numPathsPerWorker

  private[this] val lock = new Object()

  val walksFile = new File(outfile)
  val bw = new BufferedWriter(new FileWriter(walksFile))

  def processEdges() = {
    val graph: Map[Int, ArrayBuffer[(Int, Int)]] = Map()

    for ((src, rel, dst) <- edgesSc){
      val srcIdx = mapEntsIdx(src)
      val relIdx = mapEntsIdx(rel)
      val dstIdx = mapEntsIdx(dst)

      if (!graph.contains(srcIdx)){
        graph(srcIdx) = ArrayBuffer((relIdx, dstIdx))
      }else{
        graph(srcIdx) += ((relIdx, dstIdx))
      }

    }

    graph.mapValues(_.toArray)
  }


  def walk() = {

    val argsList = for (
      i <- Range(0, newWorkers, 1)
    ) yield (i, pathsPerWorker(i), walkLength)

    print("Starting pool...")

    val executor: ExecutorService = Executors.newFixedThreadPool(newWorkers)
    implicit val executionContext: ExecutionContextExecutorService = ExecutionContext.fromExecutorService(executor)

    val fut = Future.traverse(argsList)(writeWalksToDisk)
    Await.ready(fut, Duration.Inf)

    fut.onComplete {
      case Success(msg) => {
        println("* processing is over, shutting down the executor")
        executionContext.shutdown()
        bw.close
      }
      case Failure(t) =>
        {
          println("An error has ocurred in preprocessing generating random walks: " + t.getMessage + " - " + t.printStackTrace)
          executionContext.shutdown()
          bw.close
        }
    }

  }

  def writeWalksToDisk(params: (Int, Int, Int))(implicit ec: ExecutionContext): Future[Unit] = Future {
    val (index, numWalks, walkLength) = params
    println(s"+ started processing $index")

    val start = System.nanoTime() / 1000000

     for (i <- 0 until numWalks){
       val nodesR = rand.shuffle(nodesIdx)
       for (n <- nodesR){
         randomWalk(walkLength, n)
       }

     }
     
     val end = System.nanoTime() / 1000000
     val duration = (end - start)
     println(s"- finished processing $index after $duration")
  }

  def randomWalk(walkLength: Int, start: Int) = {
    val walk = Array.fill(2*walkLength-1){-1}
    walk(0) = start

    breakable {

      var i:Int = 1
      while(i < 2*walkLength-1){
        val curNode = walk(i-1)

        val lenNeighb = graph.contains(curNode) match {
          case true => graph(curNode).length
          case false => 0
        }

        if (lenNeighb >0){
          val idx = rand.nextInt(lenNeighb)
          val (nextRel, nextDst) = graph(curNode)(idx)
          walk(i) = nextRel
          walk(i+1) = nextDst

          i += 2
        }else{
          break
        }
      }
    }

    val toWrite = walk.filter(_ != -1).map(x => mapIdxEnts(x)).mkString(" ") + "\n"
    lock.synchronized{
      bw.write(toWrite)
    }
  }


    def numPathsPerWorker(): (List[Int], Int) = {

    if (numWalks <= workers) {
      val newWorkers = numWalks

      var pathsPerWorker = for (
        i <- Range(0, numWalks, 1)
      ) yield 1

      (pathsPerWorker.toList, newWorkers)
    }else{
      val newWorkers = workers
      val remainder = numWalks % workers
      var aux = workers - remainder

      val ppw = ((numWalks+aux)/workers).floor.toInt
      var pathsPerWorker = ListBuffer(ppw)

      for (i <- 0 until (workers -1)){

        pathsPerWorker += ppw
      }

      var i = 0
      while (aux > 0){
        pathsPerWorker(i%workers) =  pathsPerWorker(i%workers) - 1
        i = i+1
        aux = aux -1

      }

      (pathsPerWorker.toList, newWorkers)
    }
  }


}

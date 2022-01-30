package org.mowl.Walking

import collection.JavaConverters._
import java.io._
import java.util.{HashMap, ArrayList}
import scala.collection.mutable.{MutableList, ListBuffer, Map}
import util.control.Breaks._
import java.util.concurrent.{ExecutorService, Executors}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.concurrent.{ Await, Future }
import scala.concurrent.{ExecutionContext, ExecutionContextExecutorService}
import scala.util.{Failure, Success, Try}
import org.mowl.Edge

class WalkRDFAndOWL (
  var edges: ArrayList[Edge],
  var numWalks: Int,
  var walkLength: Int,
  var workers: Int,
  var outfile: String) {


  val edgesSc = edges.asScala.map(x => (x.src, x.rel, x.dst))
  val entities = edgesSc.map(x => List(x._1, x._2, x._3)).flatten.toSet
  val mapEntsIdx = entities.zip(Range(0, entities.size, 1)).toMap
  val mapIdxEnts = Range(0, entities.size, 1).zip(nodes).toMap
  val entsIdx = entities.map(mapEntsIdx(_))
  val nodes = edgesSc.map(x => List(x._1, x._3)).flatten.toSet
  val nodesIdx = nodes.map(mapEntsIdx(_))
  val graph = processEdges()

  val (pathsPerWorker, newWorkers) = numPathsPerWorker

  private[this] val lock = new Object()

  val walksFile = new File(outfile)
  val bw = new BufferedWriter(new FileWriter(walksFile))

  def processEdges() = {
    val graph: Map[Int, ListBuffer[(Int, Int)]] = Map()

    for ((src, rel, dst) <- edgesSc){
      val srcIdx = mapEntsIdx(src)
      val relIdx = mapEntsIdx(rel)
      val dstIdx = mapEntsIdx(dst)

      if (!graph.contains(srcIdx)){
        graph(srcIdx) = ListBuffer()
      }else{
        graph(srcIdx) += ((relIdx, dstIdx))
      }

    }

    graph
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
      case result =>
        println("* processing is over, shutting down the executor")
        executionContext.shutdown()
        bw.close
    }

  }

  def writeWalksToDisk(params: (Int, Int, Int))(implicit ec: ExecutionContext): Future[Unit] = Future {
    val (index, numWalks, walkLength) = params
    println(s"+ started processing $index")

    val start = System.nanoTime() / 1000000
    val r = scala.util.Random


     for (i <- 0 until numWalks){
       val nodesR = r.shuffle(nodesIdx)
       for (n <- nodesR){
         randomWalk(walkLength, n)
       }

     }
     
     val end = System.nanoTime() / 1000000
     val duration = (end - start)
     println(s"- finished processing $index after $duration")
  }

  def randomWalk(walkLength: Int, start: Int) = {
    var walk = MutableList(start)

    breakable {
      while(walk.length < 2*walkLength){
        var curNode = walk.last

        val lenNeighb = graph.contains(curNode) match {
          case true => graph(curNode).length
          case false => 0
        }

        val r = scala.util.Random

        if (lenNeighb >0){
          val idx = r.nextInt(lenNeighb)
          val (nextRel, nextDst) = graph(curNode)(idx)
          walk += nextRel
          walk += nextDst
        }else{
          break
        }
      }
    }

    val toWrite = walk.toList.map(x => mapIdxEnts(x).mkString(" ")) + "\n"
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

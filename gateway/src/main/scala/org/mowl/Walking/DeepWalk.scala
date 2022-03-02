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

class DeepWalk (
  var edges: ArrayList[Edge],
  var numWalks: Int,
  var walkLength: Int,
  var alpha: Float,
  var workers: Int,
  var outfile: String) {


  val edgesSc = edges.asScala.map(x => (x.src, x.dst))
  val nodes = edgesSc.map(x => List(x._1, x._2)).flatten.toSet
  val mapNodesIdx = nodes.zip(Range(0, nodes.size, 1)).toMap
  val mapIdxNodes = Range(0, nodes.size, 1).zip(nodes).toMap
  val nodesIdx = nodes.map(mapNodesIdx(_))

  val graph = processEdges()
  val rand = scala.util.Random
  val (pathsPerWorker, newWorkers) = numPathsPerWorker()

  private[this] val lock = new Object()

  val walksFile = new File(outfile)
  val bw = new BufferedWriter(new FileWriter(walksFile))


  def processEdges() = {
    val graph: Map[Int, ArrayBuffer[Int]] = Map()

    for ((src, dst) <- edgesSc){
      val srcIdx = mapNodesIdx(src)
      val dstIdx = mapNodesIdx(dst)

      if (!graph.contains(srcIdx)){
        graph(srcIdx) = ArrayBuffer(dstIdx)
      }else{
        graph(srcIdx) += dstIdx
      }
    }


    graph.mapValues(_.toArray)
  }


  def walk() = {

    val argsList = for (
      i <- Range(0, newWorkers, 1)
    ) yield (i, pathsPerWorker(i), walkLength, alpha)


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


   def writeWalksToDisk(params: (Int, Int, Int, Float))(implicit ec: ExecutionContext): Future[Unit] = Future {
     val (index, numWalks, walkLength, alpha) = params
     println(s"+ started processing $index")
     val start = System.nanoTime() / 1000000


     for (i <- 0 until numWalks){
       val nodesR = rand.shuffle(nodesIdx)
       for (n <- nodesR){
         randomWalk(walkLength, alpha, n)
       }

     }
     
     val end = System.nanoTime() / 1000000
     val duration = (end - start)
     println(s"- finished processing $index after $duration")
     
  }



  def randomWalk(walkLength: Int, alpha: Float, start: Int ) ={

    val walk = Array.fill(walkLength){-1}
    walk(0) = start

    breakable {

      var i: Int = 1
      while (i < walkLength){

        val curNode = walk(i-1)

        val lenNeighb = graph.contains(curNode) match {
          case true => graph(curNode).length
          case false => 0
        }

        if (lenNeighb > 0){
          if (rand.nextFloat >= alpha){
            val idx = rand.nextInt(lenNeighb)
            val next = graph(curNode)(idx)
            walk(i) = next
          }else{
            walk(i) = walk.head
          }
          i+=1
        }else{
          break
        }

      }

    }

    val toWrite = walk.filter(_ != -1).map(x => mapIdxNodes(x)).mkString(" ") + "\n"
    lock.synchronized {
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

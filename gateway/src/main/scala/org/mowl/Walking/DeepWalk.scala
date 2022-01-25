package org.mowl.Walking

import collection.JavaConverters._
import java.io._
import java.util.{HashMap, ArrayList}
import scala.collection.mutable.{MutableList, ListBuffer}
import util.control.Breaks._
import java.util.concurrent.{ExecutorService, Executors}
import scala.concurrent.duration.DurationLong
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.concurrent.{ Await, Future }
import scala.concurrent.Future
import scala.concurrent.{ExecutionContext, ExecutionContextExecutorService}
import scala.util.control.NonFatal
import scala.util.{Failure, Success, Try}
import scala.io.Source

class DeepWalk (
  var edges: HashMap[String, ArrayList[String]],
  var numWalks: Int,
  var walkLength: Int,
  var alpha: Float,
  var workers: Int,
  var seed: Int,
  var outfile: String) {

  private[this] val lock = new Object()

  val walksFile = new File(outfile)
  val bw = new BufferedWriter(new FileWriter(walksFile))
  val (pathsPerWorker, newWorkers) = numPathsPerWorker()

  val graph = edges.asScala.mapValues(_.asScala.toList)

  val nodes = graph.keySet.union(graph.values.flatten.toSet)

  val nodesMap_idx  = nodes.zip(Range(0, nodes.size, 1)).toMap
  val nodesMapInv_idx = Range(0, nodes.size, 1).zip(nodes).toMap

  val graphIndexed = graph.map( kv => (nodesMap_idx(kv._1), kv._2.map(x => nodesMap_idx(x))) )
  val nodes_idx = graphIndexed.keySet

  def walk(){
    val argsList = for (
      i <- Range(0, newWorkers, 1)
    ) yield (i, pathsPerWorker(i), walkLength, alpha)


    performWalks(argsList.toList)
  }

  def performWalks(argsList: List[(Int, Int, Int, Float)]) = {

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


     val r = scala.util.Random


     for (i <- 0 until numWalks){
       val nodesR = r.shuffle(nodes_idx)
       for (n <- nodesR){
         randomWalk(walkLength, alpha, n)
       }

     }
     
     val end = System.nanoTime() / 1000000
     val duration = (end - start).millis
     println(s"- finished processing $index after $duration")
     
  }



  def randomWalk(walkLength: Int, alpha: Float, start: Int ) ={

    var walk = MutableList(start)

    breakable {
      while (walk.length < walkLength){

        var curNode = walk.last

        val lenNeighb = graphIndexed.contains(curNode) match {
          case true => graphIndexed(curNode).length
          case false => 0
        }

        val r = scala.util.Random

        if (lenNeighb > 0){
          if (r.nextFloat >= alpha){
            val idx = r.nextInt(lenNeighb)
            val next = graphIndexed(curNode)(idx)
            walk += next
          }else{
            walk += walk.head
          }
        }else{
          break
        }

      }

    }

    val toWrite = walk.toList.map(x => nodesMapInv_idx(x)).mkString(" ") + "\n"
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

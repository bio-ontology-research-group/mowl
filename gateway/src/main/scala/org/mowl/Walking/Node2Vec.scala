package org.mowl.Walking

import collection.JavaConverters._
import java.io._
import java.util.{HashMap, ArrayList}
import scala.collection.mutable.{MutableList, ListBuffer, Stack, Map}
import util.control.Breaks._
import java.util.concurrent.{ExecutorService, Executors}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.concurrent.{ Await, Future }
import scala.concurrent.{ExecutionContext, ExecutionContextExecutorService}
import scala.util.{Failure, Success, Try}
import org.mowl.Edge

class Node2Vec (
  var edges: ArrayList[Edge],
  var numWalks: Int,
  var walkLength: Int,
  var p: Float,
  var q: Float,
  var workers: Int,
  var outfile: String) {


  val edgesSc = edges.asScala.map(x => (x.src, x.dst, x.weight))
  val nodes = edgesSc.map(x => List(x._1, x._2)).flatten.toSet
  val nodesSrc = edgesSc.map(_._1).toSet
  val mapNodesIdx = nodes.zip(Range(0, nodes.size, 1)).toMap
  val mapIdxNodes = Range(0, nodes.size, 1).zip(nodes).toMap
  val nodesIdx = nodes.map(mapNodesIdx(_))
  val nodesSrcIdx = nodesSrc.map(mapNodesIdx(_))
  val (graph, weights) = processEdges()

  val (pathsPerWorker, newWorkers) = numPathsPerWorker()

  var aliasNodes = Map[Int, (List[Int], List[Float])]()
  var aliasEdges = Map[(Int, Int), (List[Int], List[Float])]()

  private[this] val lock = new Object()

  val walksFile = new File(outfile)
  val bw = new BufferedWriter(new FileWriter(walksFile))


  def processEdges() = {
    val graph: Map[Int, ListBuffer[Int]] = Map()
    val weights: Map[(Int, Int), Float] = Map()
    for ((src, dst, weight) <- edgesSc){
      val srcIdx = mapNodesIdx(src)
      val dstIdx = mapNodesIdx(dst)

      if (!graph.contains(srcIdx)){
        graph(srcIdx) = ListBuffer()
      }else{
        graph(srcIdx) += dstIdx
      }

      weights((srcIdx, dstIdx)) = weight
    }


    (graph, weights)
  }

  def walk() = {

    //preprocessTransitionProbs

    val argsList = for (
      i <- Range(0, newWorkers, 1)
    ) yield (i, pathsPerWorker(i), walkLength, p, q)


    print("Starting pool...")

    val executor: ExecutorService = Executors.newFixedThreadPool(newWorkers)
    implicit val executionContext: ExecutionContextExecutorService = ExecutionContext.fromExecutorService(executor)



    println(s"+ started preprocessing probabilities...")
    val start = System.nanoTime() / 1000000

    var listsN: ListBuffer[ListBuffer[Int]] = ListBuffer.fill(newWorkers)(ListBuffer())
    for (i <- nodesSrcIdx){
      listsN(i%newWorkers) += i
    }
    val argsListN = Range(0, newWorkers, 1).zip(listsN.toList.map(_.toList))
    val futNodes = Future.traverse(argsListN)(threadNodes)

    Await.ready(futNodes, Duration.Inf)

    futNodes.onComplete {
      case result =>
        println("* processing nodes is over")
    }

    var listsE: ListBuffer[ListBuffer[(Int, Int)]] = ListBuffer.fill(newWorkers)(ListBuffer())
    val numEdges = weights.keySet.size
    for ((i, edge) <- Range(0, numEdges, 1).zip(weights.keySet.toList)){
      listsE(i%newWorkers) += edge
    }
    val argsListE = Range(0, newWorkers, 1).zip(listsE.toList.map(_.toList))
    val futEdges = Future.traverse(argsListE)(threadEdges)

    Await.ready(futEdges, Duration.Inf)

    futEdges.onComplete {
      case result =>
        println("* processing edges is over")
    }


    val end = System.nanoTime() / 1000000
    val duration = (end - start) / 1000
    println(s"- finished preprocessing probabilities  after $duration seconds")


    val futWalks = Future.traverse(argsList)(writeWalksToDisk)

    Await.ready(futWalks, Duration.Inf)

    futWalks.onComplete {
      case result =>
        println("* processing is over, shutting down the executor")
        executionContext.shutdown()
        bw.close
    }
  }


  def writeWalksToDisk(params: (Int, Int, Int, Float, Float))(implicit ec: ExecutionContext): Future[Unit] = Future {

    val (index, numWalks, walkLength, p, q) = params

    println(s"+ started processing $index")
    val start = System.nanoTime() / 1000000


    val r = scala.util.Random

    for (i <- 0 until numWalks){
      val nodesR = r.shuffle(nodesIdx)
      for (n <- nodesR){
        randomWalk(walkLength, p, q, n)
      }

    }
     
    val end = System.nanoTime() / 1000000
    val duration = (end - start)
    println(s"- finished processing $index after $duration")
  }



  def randomWalk(walkLength: Int, p: Float, q: Float, start: Int ) = {

    var walk = MutableList(start)

    breakable {

      while(walk.length < walkLength){
        val curNode = walk.last

        val curNbrs = graph.contains(curNode) match {
          case true => graph(curNode).sorted
          case false => Nil
        }

        if (curNbrs.length > 0) {

          if (walk.length == 1){
            val (idx1, idx2) = aliasNodes(curNode)
            walk += curNbrs(aliasDraw(idx1, idx2))

          }else {
            val prevNode = walk.init.last
            val (idx1, idx2) = aliasEdges((prevNode, curNode))
            walk += curNbrs(aliasDraw(idx1, idx2))

          }
        }else{
          break
        }

      }

    }

    val toWrite = walk.toList.map(x => mapIdxNodes(x)).mkString(" ") + "\n"
    lock.synchronized {
      bw.write(toWrite)
    }

  }



  def getAliasEdge(src: Int, dst: Int) = {

    var unnormalizedProbs: MutableList[Float] = MutableList()

    if (graph.contains(dst)){

      for (dstNbr <- graph(dst).sorted) {
        val prob = weights((dst, dstNbr))

        if (dstNbr == src){
          unnormalizedProbs += prob/p
        }else if( weights.contains((dstNbr, src))){
          unnormalizedProbs += prob
        }else{
          unnormalizedProbs += prob/q
        }
      }

      val normConst = unnormalizedProbs.sum
      val normalizedProbs = unnormalizedProbs.map(x => x/normConst)

      aliasSetup(normalizedProbs.toList)

    }else{
      aliasSetup(Nil)
    }

  }



  def threadNodes(params: (Int, List[Int]))(implicit ec: ExecutionContext): Future[Unit] = Future {

    val (idx, indices) = params
    val l = indices.length
    println(s"Thread $idx, nodes to process $l")
    for (i <- indices){
      val node = i
      val unnormalizedProbs = graph(node).sorted.map(nbr => weights((node, nbr)))
      val normConst = unnormalizedProbs.sum
      val normalizedProbs = unnormalizedProbs.map(x => x/normConst).toList

      lock.synchronized {
        aliasNodes += (node -> aliasSetup(normalizedProbs))
      }

    }

  }

  def threadEdges(params:  (Int, List[(Int, Int)]))(implicit ec: ExecutionContext): Future[Unit] = Future {

    val (idx, edges) = params
    val l = edges.length
    println(s"Thread $idx, edges to process $l")

    for ((src, dst) <- edges){

      val alias = getAliasEdge(src, dst)
      lock.synchronized {
        aliasEdges += ((src, dst) -> alias)
      }

    }

  }

  //////////////////////////////////////////
  //https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/


  def aliasSetup(probs: List[Float]) = {

    val K = probs.length
    val q: ListBuffer[Float] = ListBuffer.fill(K)(0)
    val J: ListBuffer[Int] = ListBuffer.fill(K)(0)

    val smaller = Stack[Int]()
    val larger = Stack[Int]()

    for ((kk, prob) <- Range(0, K, 1).zip(probs)){

      q(kk) = K*prob

      if (q(kk) < 1){
        smaller.push(kk)
      }else {
        larger.push(kk)
      }
    }

    while (smaller.length > 0 && larger.length > 0){
      val small = smaller.pop
      val large = larger.pop

      J(small) = large
      q(large) = q(large) + q(small) - 1

      if (q(large) < 1){
        smaller.push(large)
      }else{
        larger.push(large)
      }
    }

    (J.toList, q.toList)

  }

  def aliasDraw(J: List[Int], q: List[Float]): Int  = {

    val r = scala.util.Random
    val K = J.length
    val kk = (r.nextFloat * K).floor.toInt

    if (r.nextFloat < q(kk)){
      kk
    }else{
      J(kk)
    }
  }


  /////////////////////////////////////////







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



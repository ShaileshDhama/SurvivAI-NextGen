package survivai.services

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.{Future, ExecutionContext}
import scala.concurrent.ExecutionContext.Implicits.global
import survivai.models.Analysis._

object AnalysisService {
  private val baseUrl = "/api/analyses"
  
  // Get all analyses with optional filters
  def getAnalyses(filters: Option[Filters] = None): Future[Seq[Analysis]] = {
    val params = filters.map { f =>
      val jsObj = js.Dynamic.literal()
      
      f.searchTerm.foreach(jsObj.searchTerm = _)
      f.datasetId.foreach(jsObj.datasetId = _)
      f.status.foreach(s => jsObj.status = Status.toString(s))
      f.analysisType.foreach(t => jsObj.analysisType = AnalysisType.toString(t))
      f.sortBy.foreach(jsObj.sortBy = _)
      f.sortDirection.foreach(jsObj.sortDirection = _)
      
      jsObj.asInstanceOf[js.Object]
    }.getOrElse(null)
    
    HttpClient.get[js.Array[js.Dynamic]](baseUrl, params)
      .map(_.map(fromJS).toSeq)
  }
  
  // Get a specific analysis by ID
  def getAnalysis(id: String): Future[Analysis] = {
    HttpClient.get[js.Dynamic](s"$baseUrl/$id")
      .map(fromJS)
  }
  
  // Create a new analysis
  def createAnalysis(payload: CreateAnalysisPayload): Future[Analysis] = {
    val jsPayload = js.Dynamic.literal(
      name = payload.name,
      datasetId = payload.datasetId,
      timeColumn = payload.timeColumn,
      eventColumn = payload.eventColumn,
      analysisType = payload.analysisType,
      covariates = payload.covariates.toArray,
      parameters = payload.parameters
    )
    
    payload.description.foreach(jsPayload.description = _)
    
    HttpClient.post[js.Dynamic](baseUrl, jsPayload.asInstanceOf[js.Object])
      .map(fromJS)
  }
  
  // Update an existing analysis
  def updateAnalysis(id: String, analysis: Analysis): Future[Analysis] = {
    HttpClient.put[js.Dynamic](s"$baseUrl/$id", analysis.toJS)
      .map(fromJS)
  }
  
  // Delete an analysis
  def deleteAnalysis(id: String): Future[Boolean] = {
    HttpClient.delete[js.Dynamic](s"$baseUrl/$id")
      .map(_ => true)
      .recover { case _ => false }
  }
  
  // Run an analysis
  def runAnalysis(id: String): Future[Analysis] = {
    HttpClient.post[js.Dynamic](s"$baseUrl/$id/run", null)
      .map(fromJS)
  }
  
  // Get analysis results
  def getResults(id: String): Future[js.Dynamic] = {
    HttpClient.get[js.Dynamic](s"$baseUrl/$id/results")
  }
  
  // Get survival curves for an analysis
  def getSurvivalCurves(id: String): Future[js.Array[js.Array[SurvivalPoint]]] = {
    HttpClient.get[js.Array[js.Array[js.Dynamic]]](s"$baseUrl/$id/survival-curves")
      .map { curves =>
        curves.map { curve =>
          curve.map { point =>
            val upper = if (js.isUndefined(point.upper) || point.upper == null) None
                       else Some(point.upper.asInstanceOf[Double])
            
            val lower = if (js.isUndefined(point.lower) || point.lower == null) None
                       else Some(point.lower.asInstanceOf[Double])
            
            SurvivalPoint(
              time = point.time.asInstanceOf[Double],
              survival = point.survival.asInstanceOf[Double],
              upper = upper,
              lower = lower
            )
          }
        }
      }
  }
}

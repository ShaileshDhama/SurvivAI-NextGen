package survivai.services

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.{Future, ExecutionContext}
import scala.concurrent.ExecutionContext.Implicits.global
import org.scalajs.dom.FormData
import survivai.models.Dataset._

object DatasetService {
  private val baseUrl = "/api/datasets"
  
  // Get all datasets with optional filters
  def getDatasets(filters: Option[Filters] = None): Future[Seq[Dataset]] = {
    val params = filters.map { f =>
      val jsObj = js.Dynamic.literal()
      
      f.searchTerm.foreach(jsObj.searchTerm = _)
      f.sortBy.foreach(jsObj.sortBy = _)
      f.sortDirection.foreach(jsObj.sortDirection = _)
      
      jsObj.asInstanceOf[js.Object]
    }.getOrElse(null)
    
    HttpClient.get[js.Array[js.Dynamic]](baseUrl, params)
      .map(_.map(fromJS).toSeq)
  }
  
  // Get a specific dataset by ID
  def getDataset(id: String): Future[Dataset] = {
    HttpClient.get[js.Dynamic](s"$baseUrl/$id")
      .map(fromJS)
  }
  
  // Upload a new dataset
  def uploadDataset(file: org.scalajs.dom.File, name: String, description: Option[String] = None): Future[Dataset] = {
    val formData = new FormData()
    formData.append("file", file)
    formData.append("name", name)
    description.foreach(d => formData.append("description", d))
    
    val config = js.Dynamic.literal(
      method = "post",
      url = baseUrl,
      data = formData,
      headers = js.Dynamic.literal("Content-Type" = "multipart/form-data")
    )
    
    // Use progress tracking
    val onUploadProgress = js.Function1 { (progressEvent: js.Dynamic) =>
      val percentCompleted = Math.round((progressEvent.loaded.asInstanceOf[Double] * 100) / progressEvent.total.asInstanceOf[Double])
      js.Dynamic.global.console.log(s"Upload progress: $percentCompleted%")
    }
    
    config.onUploadProgress = onUploadProgress
    
    HttpClient.request[js.Dynamic](config.asInstanceOf[js.Dynamic])
      .map(fromJS)
  }
  
  // Delete a dataset
  def deleteDataset(id: String): Future[Boolean] = {
    HttpClient.delete[js.Dynamic](s"$baseUrl/$id")
      .map(_ => true)
      .recover { case _ => false }
  }
  
  // Get dataset schema (column information)
  def getSchema(id: String): Future[Seq[Column]] = {
    HttpClient.get[js.Array[js.Dynamic]](s"$baseUrl/$id/schema")
      .map(_.map(columnFromJS).toSeq)
  }
  
  // Get dataset summary statistics
  def getSummary(id: String): Future[Summary] = {
    HttpClient.get[js.Dynamic](s"$baseUrl/$id/summary")
      .map(summaryFromJS)
  }
  
  // Get dataset preview (first few rows)
  def getPreview(id: String, rows: Int = 10): Future[js.Array[js.Object]] = {
    val params = js.Dynamic.literal(rows = rows).asInstanceOf[js.Object]
    HttpClient.get[js.Array[js.Object]](s"$baseUrl/$id/preview", params)
  }
  
  // Update dataset metadata
  def updateDataset(id: String, name: String, description: Option[String]): Future[Dataset] = {
    val jsPayload = js.Dynamic.literal(name = name)
    description.foreach(jsPayload.description = _)
    
    HttpClient.put[js.Dynamic](s"$baseUrl/$id", jsPayload.asInstanceOf[js.Object])
      .map(fromJS)
  }
}

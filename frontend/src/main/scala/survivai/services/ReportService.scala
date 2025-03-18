package survivai.services

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.{Future, Promise}
import scala.util.{Success, Failure}
import org.scalajs.dom
import survivai.models.Report

object ReportService {
  private val apiBase = "http://localhost:3001/api/v1"
  
  /**
   * Get all reports
   * @return Future containing a sequence of reports
   */
  def getReports(): Future[Seq[Report.Report]] = {
    val promise = Promise[Seq[Report.Report]]()
    
    dom.fetch(s"$apiBase/reports")
      .`then`[js.Dynamic](response => {
        if (!response.ok.asInstanceOf[Boolean]) {
          throw js.Error(s"Failed to fetch reports: ${response.statusText.asInstanceOf[String]}")
        }
        response.json().asInstanceOf[js.Promise[js.Dynamic]]
      })
      .`then`[Unit](data => {
        val reportsArray = data.asInstanceOf[js.Array[js.Dynamic]]
        val reports = reportsArray.map(Report.fromJS).toSeq
        promise.success(reports)
      })
      .`catch`(error => {
        console.error("Error fetching reports:", error)
        promise.failure(new Exception(s"Error fetching reports: ${error.toString}"))
      })
    
    promise.future
  }
  
  /**
   * Get a specific report by ID
   * @param id the report ID
   * @return Future containing the report
   */
  def getReport(id: String): Future[Report.Report] = {
    val promise = Promise[Report.Report]()
    
    dom.fetch(s"$apiBase/reports/$id")
      .`then`[js.Dynamic](response => {
        if (!response.ok.asInstanceOf[Boolean]) {
          throw js.Error(s"Failed to fetch report: ${response.statusText.asInstanceOf[String]}")
        }
        response.json().asInstanceOf[js.Promise[js.Dynamic]]
      })
      .`then`[Unit](data => {
        val report = Report.fromJS(data)
        promise.success(report)
      })
      .`catch`(error => {
        console.error("Error fetching report:", error)
        promise.failure(new Exception(s"Error fetching report: ${error.toString}"))
      })
    
    promise.future
  }
  
  /**
   * Create a new report
   * @param reportData the report creation payload
   * @return Future containing the created report
   */
  def createReport(reportData: Report.CreateReportPayload): Future[Report.Report] = {
    val promise = Promise[Report.Report]()
    
    val requestBody = js.Dynamic.literal(
      title = reportData.title,
      description = reportData.description.getOrElse(null),
      analysisId = reportData.analysisId,
      includeSections = reportData.includeSections.toArray
    )
    
    val fetchOptions = js.Dynamic.literal(
      method = "POST",
      headers = js.Dynamic.literal(
        "Content-Type" = "application/json"
      ),
      body = js.JSON.stringify(requestBody.asInstanceOf[js.Object])
    )
    
    dom.fetch(s"$apiBase/reports", fetchOptions.asInstanceOf[RequestInit])
      .`then`[js.Dynamic](response => {
        if (!response.ok.asInstanceOf[Boolean]) {
          throw js.Error(s"Failed to create report: ${response.statusText.asInstanceOf[String]}")
        }
        response.json().asInstanceOf[js.Promise[js.Dynamic]]
      })
      .`then`[Unit](data => {
        val report = Report.fromJS(data)
        promise.success(report)
      })
      .`catch`(error => {
        console.error("Error creating report:", error)
        promise.failure(new Exception(s"Error creating report: ${error.toString}"))
      })
    
    promise.future
  }
  
  /**
   * Delete a report
   * @param id the report ID to delete
   * @return Future containing a boolean indicating success
   */
  def deleteReport(id: String): Future[Boolean] = {
    val promise = Promise[Boolean]()
    
    val fetchOptions = js.Dynamic.literal(
      method = "DELETE"
    )
    
    dom.fetch(s"$apiBase/reports/$id", fetchOptions.asInstanceOf[RequestInit])
      .`then`[Unit](response => {
        if (!response.ok.asInstanceOf[Boolean]) {
          throw js.Error(s"Failed to delete report: ${response.statusText.asInstanceOf[String]}")
        }
        promise.success(true)
      })
      .`catch`(error => {
        console.error("Error deleting report:", error)
        promise.failure(new Exception(s"Error deleting report: ${error.toString}"))
      })
    
    promise.future
  }
  
  /**
   * Send a question to the report chatbot
   * @param question the user's question
   * @param reportId optional report ID for context
   * @param modelId the model ID for context
   * @return Future containing the chatbot response
   */
  def askChatbot(question: String, reportId: Option[String], modelId: String): Future[String] = {
    val promise = Promise[String]()
    
    val requestBody = js.Dynamic.literal(
      question = question,
      model_id = modelId
    )
    
    reportId.foreach(id => requestBody.report_id = id)
    
    val fetchOptions = js.Dynamic.literal(
      method = "POST",
      headers = js.Dynamic.literal(
        "Content-Type" = "application/json"
      ),
      body = js.JSON.stringify(requestBody.asInstanceOf[js.Object])
    )
    
    dom.fetch(
      s"$apiBase/reports/chatbot/question", 
      fetchOptions.asInstanceOf[RequestInit]
    )
      .`then`[js.Dynamic](response => {
        if (!response.ok.asInstanceOf[Boolean]) {
          throw js.Error(s"Failed to get chatbot response: ${response.statusText.asInstanceOf[String]}")
        }
        response.json().asInstanceOf[js.Promise[js.Dynamic]]
      })
      .`then`[Unit](data => {
        val response = data.response.asInstanceOf[String]
        promise.success(response)
      })
      .`catch`(error => {
        console.error("Error getting chatbot response:", error)
        promise.failure(new Exception(s"Error getting chatbot response: ${error.toString}"))
      })
    
    promise.future
  }
}

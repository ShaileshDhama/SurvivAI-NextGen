package survivai.services

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.{Future, Promise}
import scala.util.{Success, Failure}
import scala.concurrent.ExecutionContext.Implicits.global
import survivai.models.Visualization
import org.scalajs.dom.experimental.HttpMethod
import org.scalajs.dom.experimental.Fetch
import org.scalajs.dom.experimental.RequestInit
import org.scalajs.dom.experimental.Response

object VisualizationService {
  private val apiUrl = "http://localhost:3001/api"
  
  // Get all visualizations
  def getVisualizations(): Future[Seq[Visualization.Visualization]] = {
    val promise = Promise[Seq[Visualization.Visualization]]()
    
    val request = Fetch.fetch(
      s"$apiUrl/visualizations", 
      new RequestInit {
        method = HttpMethod.GET
        headers = js.Dictionary("Content-Type" -> "application/json")
      }
    )
    
    request.`then`[Unit](
      (response: Response) => {
        if (response.ok) {
          response.json().`then`[Unit](
            (data: js.Dynamic) => {
              // Extract visualizations from response
              val visualizations = data.asInstanceOf[js.Array[js.Dynamic]]
                .map(viz => Visualization.fromJs(viz))
                .toSeq
              
              promise.success(visualizations)
              ()
            },
            (error: js.Error) => {
              promise.failure(new Exception(s"Failed to parse response: ${error.message}"))
              ()
            }
          )
        } else {
          promise.failure(new Exception(s"Failed to fetch visualizations: ${response.statusText}"))
        }
        ()
      },
      (error: js.Error) => {
        promise.failure(new Exception(s"Request failed: ${error.message}"))
        ()
      }
    )
    
    promise.future
  }
  
  // Get a specific visualization by ID
  def getVisualization(id: String): Future[Visualization.Visualization] = {
    val promise = Promise[Visualization.Visualization]()
    
    val request = Fetch.fetch(
      s"$apiUrl/visualizations/$id", 
      new RequestInit {
        method = HttpMethod.GET
        headers = js.Dictionary("Content-Type" -> "application/json")
      }
    )
    
    request.`then`[Unit](
      (response: Response) => {
        if (response.ok) {
          response.json().`then`[Unit](
            (data: js.Dynamic) => {
              promise.success(Visualization.fromJs(data))
              ()
            },
            (error: js.Error) => {
              promise.failure(new Exception(s"Failed to parse response: ${error.message}"))
              ()
            }
          )
        } else {
          promise.failure(new Exception(s"Failed to fetch visualization: ${response.statusText}"))
        }
        ()
      },
      (error: js.Error) => {
        promise.failure(new Exception(s"Request failed: ${error.message}"))
        ()
      }
    )
    
    promise.future
  }
  
  // Create a new visualization
  def createVisualization(visualization: Visualization.VisualizationCreate): Future[Visualization.Visualization] = {
    val promise = Promise[Visualization.Visualization]()
    
    // Convert to JS object format for API
    val jsData = js.Dynamic.literal(
      title = visualization.title,
      analysisId = visualization.analysisId,
      visualizationType = Visualization.VisualizationType.toString(visualization.visualizationType),
      config = visualization.config
    )
    
    // Add optional fields
    visualization.description.foreach(jsData.description = _)
    
    val request = Fetch.fetch(
      s"$apiUrl/visualizations", 
      new RequestInit {
        method = HttpMethod.POST
        headers = js.Dictionary("Content-Type" -> "application/json")
        body = js.JSON.stringify(jsData)
      }
    )
    
    request.`then`[Unit](
      (response: Response) => {
        if (response.ok) {
          response.json().`then`[Unit](
            (data: js.Dynamic) => {
              promise.success(Visualization.fromJs(data))
              ()
            },
            (error: js.Error) => {
              promise.failure(new Exception(s"Failed to parse response: ${error.message}"))
              ()
            }
          )
        } else {
          // Handle error response
          response.text().`then`[Unit](
            (errorText: String) => {
              promise.failure(new Exception(s"Failed to create visualization: $errorText"))
              ()
            },
            (error: js.Error) => {
              promise.failure(new Exception(s"Failed to create visualization: ${response.statusText}"))
              ()
            }
          )
        }
        ()
      },
      (error: js.Error) => {
        promise.failure(new Exception(s"Request failed: ${error.message}"))
        ()
      }
    )
    
    promise.future
  }
  
  // Update an existing visualization
  def updateVisualization(visualization: Visualization.VisualizationUpdate): Future[Visualization.Visualization] = {
    val promise = Promise[Visualization.Visualization]()
    
    // Convert to JS object format for API, only including defined fields
    val jsData = js.Dynamic.literal()
    
    if (js.typeOf(visualization.title) != "undefined") {
      jsData.title = visualization.title
    }
    
    if (js.typeOf(visualization.description) != "undefined") {
      jsData.description = visualization.description
    }
    
    if (js.typeOf(visualization.visualizationType) != "undefined") {
      jsData.visualizationType = Visualization.VisualizationType.toString(visualization.visualizationType.asInstanceOf[Visualization.VisualizationType])
    }
    
    if (js.typeOf(visualization.config) != "undefined") {
      jsData.config = visualization.config
    }
    
    val request = Fetch.fetch(
      s"$apiUrl/visualizations/${visualization.id}", 
      new RequestInit {
        method = HttpMethod.PUT
        headers = js.Dictionary("Content-Type" -> "application/json")
        body = js.JSON.stringify(jsData)
      }
    )
    
    request.`then`[Unit](
      (response: Response) => {
        if (response.ok) {
          response.json().`then`[Unit](
            (data: js.Dynamic) => {
              promise.success(Visualization.fromJs(data))
              ()
            },
            (error: js.Error) => {
              promise.failure(new Exception(s"Failed to parse response: ${error.message}"))
              ()
            }
          )
        } else {
          promise.failure(new Exception(s"Failed to update visualization: ${response.statusText}"))
        }
        ()
      },
      (error: js.Error) => {
        promise.failure(new Exception(s"Request failed: ${error.message}"))
        ()
      }
    )
    
    promise.future
  }
  
  // Delete a visualization
  def deleteVisualization(id: String): Future[Unit] = {
    val promise = Promise[Unit]()
    
    val request = Fetch.fetch(
      s"$apiUrl/visualizations/$id", 
      new RequestInit {
        method = HttpMethod.DELETE
        headers = js.Dictionary("Content-Type" -> "application/json")
      }
    )
    
    request.`then`[Unit](
      (response: Response) => {
        if (response.ok) {
          promise.success(())
        } else {
          promise.failure(new Exception(s"Failed to delete visualization: ${response.statusText}"))
        }
        ()
      },
      (error: js.Error) => {
        promise.failure(new Exception(s"Request failed: ${error.message}"))
        ()
      }
    )
    
    promise.future
  }
}

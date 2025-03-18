package survivai.services

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.{Future, Promise}
import scala.util.{Success, Failure}
import scala.concurrent.ExecutionContext.Implicits.global
import survivai.models.Model
import org.scalajs.dom.experimental.HttpMethod
import org.scalajs.dom.experimental.Fetch
import org.scalajs.dom.experimental.RequestInit
import org.scalajs.dom.experimental.Response

@JSExportTopLevel("ModelService")
object ModelService {
  private val apiUrl = "http://localhost:3001/api"
  
  /**
   * Get all models with optional filters
   */
  def getModels(filters: Option[js.Object] = None): Future[Seq[Model.Model]] = {
    val promise = Promise[Seq[Model.Model]]()
    
    // Create URL with query params if filters are provided
    val url = filters.map { f =>
      val queryParams = js.Object.entries(f)
        .map { case entry => 
          s"${entry.asInstanceOf[js.Array[String]](0)}=${entry.asInstanceOf[js.Array[String]](1)}"
        }
        .mkString("&")
      s"$apiUrl/models?$queryParams"
    }.getOrElse(s"$apiUrl/models")
    
    val request = Fetch.fetch(
      url, 
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
              // Extract models from response
              val models = data.asInstanceOf[js.Array[js.Dynamic]]
                .map(model => Model.fromJs(model))
                .toSeq
              
              promise.success(models)
              ()
            },
            (error: js.Error) => {
              promise.failure(new Exception(s"Failed to parse response: ${error.message}"))
              ()
            }
          )
        } else {
          promise.failure(new Exception(s"Failed to fetch models: ${response.statusText}"))
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
  
  /**
   * Get a specific model by ID
   */
  def getModel(id: String): Future[Model.Model] = {
    val promise = Promise[Model.Model]()
    
    val request = Fetch.fetch(
      s"$apiUrl/models/$id", 
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
              promise.success(Model.fromJs(data))
              ()
            },
            (error: js.Error) => {
              promise.failure(new Exception(s"Failed to parse response: ${error.message}"))
              ()
            }
          )
        } else {
          promise.failure(new Exception(s"Failed to fetch model: ${response.statusText}"))
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
  
  /**
   * Create a new model
   */
  def createModel(model: Model.ModelCreate): Future[Model.Model] = {
    val promise = Promise[Model.Model]()
    
    // Convert to JS object format for API
    val jsData = js.Dynamic.literal(
      name = model.name,
      analysisId = model.analysisId,
      modelType = Model.ModelType.toString(model.modelType),
      parameters = model.parameters
    )
    
    // Add optional fields
    model.description.foreach(jsData.description = _)
    
    val request = Fetch.fetch(
      s"$apiUrl/models", 
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
              promise.success(Model.fromJs(data))
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
              promise.failure(new Exception(s"Failed to create model: $errorText"))
              ()
            },
            (error: js.Error) => {
              promise.failure(new Exception(s"Failed to create model: ${response.statusText}"))
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
  
  /**
   * Update an existing model
   */
  def updateModel(model: Model.ModelUpdate): Future[Model.Model] = {
    val promise = Promise[Model.Model]()
    
    // Convert to JS object format for API, only including defined fields
    val jsData = js.Dynamic.literal()
    
    if (js.typeOf(model.name) != "undefined") {
      jsData.name = model.name
    }
    
    if (js.typeOf(model.description) != "undefined") {
      jsData.description = model.description
    }
    
    if (js.typeOf(model.parameters) != "undefined") {
      jsData.parameters = model.parameters
    }
    
    val request = Fetch.fetch(
      s"$apiUrl/models/${model.id}", 
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
              promise.success(Model.fromJs(data))
              ()
            },
            (error: js.Error) => {
              promise.failure(new Exception(s"Failed to parse response: ${error.message}"))
              ()
            }
          )
        } else {
          response.text().`then`[Unit](
            (errorText: String) => {
              promise.failure(new Exception(s"Failed to update model: $errorText"))
              ()
            },
            (error: js.Error) => {
              promise.failure(new Exception(s"Failed to update model: ${response.statusText}"))
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
  
  /**
   * Delete a model
   */
  def deleteModel(id: String): Future[Boolean] = {
    val promise = Promise[Boolean]()
    
    val request = Fetch.fetch(
      s"$apiUrl/models/$id", 
      new RequestInit {
        method = HttpMethod.DELETE
        headers = js.Dictionary("Content-Type" -> "application/json")
      }
    )
    
    request.`then`[Unit](
      (response: Response) => {
        if (response.ok) {
          promise.success(true)
        } else {
          response.text().`then`[Unit](
            (errorText: String) => {
              promise.failure(new Exception(s"Failed to delete model: $errorText"))
              ()
            },
            (error: js.Error) => {
              promise.failure(new Exception(s"Failed to delete model: ${response.statusText}"))
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
  
  /**
   * Get predictions from a model
   */
  def getPredictions(modelId: String, inputData: js.Object): Future[js.Object] = {
    val promise = Promise[js.Object]()
    
    val request = Fetch.fetch(
      s"$apiUrl/models/$modelId/predict", 
      new RequestInit {
        method = HttpMethod.POST
        headers = js.Dictionary("Content-Type" -> "application/json")
        body = js.JSON.stringify(inputData)
      }
    )
    
    request.`then`[Unit](
      (response: Response) => {
        if (response.ok) {
          response.json().`then`[Unit](
            (data: js.Dynamic) => {
              promise.success(data.asInstanceOf[js.Object])
              ()
            },
            (error: js.Error) => {
              promise.failure(new Exception(s"Failed to parse response: ${error.message}"))
              ()
            }
          )
        } else {
          response.text().`then`[Unit](
            (errorText: String) => {
              promise.failure(new Exception(s"Failed to get predictions: $errorText"))
              ()
            },
            (error: js.Error) => {
              promise.failure(new Exception(s"Failed to get predictions: ${response.statusText}"))
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
}

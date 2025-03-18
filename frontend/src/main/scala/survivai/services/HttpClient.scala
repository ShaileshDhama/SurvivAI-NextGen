package survivai.services

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.{Future, Promise}
import scala.util.{Success, Failure}

object HttpClient {
  @js.native
  @JSImport("axios", JSImport.Default)
  private val axios: js.Dynamic = js.native
  
  private val baseURL = "/api"
  
  private val defaultConfig = js.Dynamic.literal(
    baseURL = baseURL,
    headers = js.Dynamic.literal(
      "Content-Type" = "application/json"
    )
  )
  
  private val axiosInstance = axios.create(defaultConfig)
  
  // Set auth token for requests
  def setAuthToken(token: String): Unit = {
    axiosInstance.defaults.headers.common.Authorization = s"Bearer $token"
  }
  
  // Remove auth token
  def removeAuthToken(): Unit = {
    js.Dynamic.global.delete(axiosInstance.defaults.headers.common.Authorization)
  }
  
  // Generic request method
  private def request[T](config: js.Dynamic): Future[T] = {
    val promise = Promise[T]()
    
    axiosInstance(config).`then`[Unit](
      (response: js.Dynamic) => {
        promise.success(response.data.asInstanceOf[T])
        (): Unit
      },
      (error: js.Dynamic) => {
        promise.failure(new Exception(error.message.asInstanceOf[String]))
        (): Unit
      }
    )
    
    promise.future
  }
  
  // Convenience methods for different HTTP methods
  def get[T](url: String, params: js.Object = null): Future[T] = {
    val config = js.Dynamic.literal(method = "get", url = url)
    if (params != null) config.params = params
    request[T](config)
  }
  
  def post[T](url: String, data: js.Object): Future[T] = {
    val config = js.Dynamic.literal(method = "post", url = url, data = data)
    request[T](config)
  }
  
  def put[T](url: String, data: js.Object): Future[T] = {
    val config = js.Dynamic.literal(method = "put", url = url, data = data)
    request[T](config)
  }
  
  def delete[T](url: String): Future[T] = {
    val config = js.Dynamic.literal(method = "delete", url = url)
    request[T](config)
  }
}

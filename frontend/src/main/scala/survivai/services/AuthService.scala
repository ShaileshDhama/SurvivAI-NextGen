package survivai.services

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.{Future, ExecutionContext}
import scala.concurrent.ExecutionContext.Implicits.global

object AuthService {
  // User model
  case class User(id: String, username: String, email: String)
  
  // Response models
  case class LoginResponse(token: String, user: User)
  case class AuthResponse(success: Boolean, message: String, user: Option[User] = None)
  
  // JavaScript conversions
  implicit class UserOps(user: User) {
    def toJS: js.Object = {
      js.Dynamic.literal(
        id = user.id,
        username = user.username,
        email = user.email
      ).asInstanceOf[js.Object]
    }
  }
  
  implicit def userFromJS(obj: js.Dynamic): User = {
    User(
      id = obj.id.asInstanceOf[String],
      username = obj.username.asInstanceOf[String],
      email = obj.email.asInstanceOf[String]
    )
  }
  
  // Methods
  def login(email: String, password: String): Future[AuthResponse] = {
    val credentials = js.Dynamic.literal(
      email = email,
      password = password
    )
    
    HttpClient.post[js.Dynamic]("/auth/login", credentials.asInstanceOf[js.Object])
      .map { response =>
        val token = response.token.asInstanceOf[String]
        val user = userFromJS(response.user)
        
        // Store token
        js.Dynamic.global.localStorage.setItem("token", token)
        HttpClient.setAuthToken(token)
        
        AuthResponse(success = true, message = "Login successful", Some(user))
      }
      .recover {
        case e: Exception => AuthResponse(success = false, message = e.getMessage)
      }
  }
  
  def logout(): Unit = {
    js.Dynamic.global.localStorage.removeItem("token")
    HttpClient.removeAuthToken()
  }
  
  def getCurrentUser(): Future[Option[User]] = {
    val token = js.Dynamic.global.localStorage.getItem("token").asInstanceOf[String]
    
    if (token == null || token.isEmpty) {
      Future.successful(None)
    } else {
      HttpClient.setAuthToken(token)
      HttpClient.get[js.Dynamic]("/auth/me")
        .map { userData =>
          Some(userFromJS(userData))
        }
        .recover { case _ => None }
    }
  }
  
  def isAuthenticated(): Boolean = {
    val token = js.Dynamic.global.localStorage.getItem("token").asInstanceOf[String]
    token != null && token.nonEmpty
  }
}

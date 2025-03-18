package survivai.contexts

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import survivai.bindings.ReactBindings.*
import survivai.services.AuthService
import survivai.services.AuthService.User

object AuthContext {
  case class AuthState(
    isAuthenticated: Boolean = false,
    user: Option[User] = None,
    loading: Boolean = true,
    error: Option[String] = None
  )
  
  sealed trait AuthAction
  case class LoginSuccess(user: User) extends AuthAction
  case class LoginFailure(error: String) extends AuthAction
  case object Logout extends AuthAction
  case object SetLoading extends AuthAction
  case class SetUser(user: Option[User]) extends AuthAction
  
  private val initialState = AuthState()
  private val authContext = React.createContext[js.Dynamic](null)
  
  private def authReducer(state: AuthState, action: AuthAction): AuthState = action match {
    case LoginSuccess(user) => 
      state.copy(isAuthenticated = true, user = Some(user), loading = false, error = None)
    case LoginFailure(error) => 
      state.copy(isAuthenticated = false, user = None, loading = false, error = Some(error))
    case Logout => 
      state.copy(isAuthenticated = false, user = None, loading = false, error = None)
    case SetLoading => 
      state.copy(loading = true)
    case SetUser(user) => 
      state.copy(isAuthenticated = user.isDefined, user = user, loading = false)
  }
  
  def Provider(children: Element): Element = {
    FC {
      val jsReducer = js.Function2 { (state: js.Dynamic, action: js.Dynamic) =>
        val scalaState = AuthState(
          isAuthenticated = state.isAuthenticated.asInstanceOf[Boolean],
          user = Option(state.user.asInstanceOf[js.Dynamic])
            .filterNot(_ == js.undefined)
            .map(u => AuthService.userFromJS(u)),
          loading = state.loading.asInstanceOf[Boolean],
          error = Option(state.error.asInstanceOf[js.Any])
            .filterNot(_ == js.undefined)
            .map(_.toString)
        )
        
        val actionType = action.`type`.asInstanceOf[String]
        val newState = actionType match {
          case "LOGIN_SUCCESS" => authReducer(scalaState, LoginSuccess(AuthService.userFromJS(action.payload)))
          case "LOGIN_FAILURE" => authReducer(scalaState, LoginFailure(action.payload.asInstanceOf[String]))
          case "LOGOUT" => authReducer(scalaState, Logout)
          case "SET_LOADING" => authReducer(scalaState, SetLoading)
          case "SET_USER" => 
            val userOpt = Option(action.payload.asInstanceOf[js.Dynamic])
              .filterNot(_ == js.undefined)
              .map(u => AuthService.userFromJS(u))
            authReducer(scalaState, SetUser(userOpt))
          case _ => scalaState
        }
        
        js.Dynamic.literal(
          isAuthenticated = newState.isAuthenticated,
          user = newState.user.map(_.toJS).getOrElse(null),
          loading = newState.loading,
          error = newState.error.getOrElse(null)
        )
      }
      
      val jsInitialState = js.Dynamic.literal(
        isAuthenticated = initialState.isAuthenticated,
        user = initialState.user.map(_.toJS).getOrElse(null),
        loading = initialState.loading,
        error = initialState.error.getOrElse(null)
      )
      
      val stateAndDispatch = React.useReducer(jsReducer, jsInitialState)
      val state = stateAndDispatch(0).asInstanceOf[js.Dynamic]
      val dispatch = stateAndDispatch(1).asInstanceOf[js.Function1[js.Dynamic, Unit]]
      
      // Check if user is logged in on load
      React.useEffect(() => {
        if (AuthService.isAuthenticated()) {
          AuthService.getCurrentUser().foreach { userOpt =>
            val action = js.Dynamic.literal(
              `type` = "SET_USER",
              payload = userOpt.map(_.toJS).getOrElse(null)
            )
            dispatch(action)
          }
        } else {
          val action = js.Dynamic.literal(
            `type` = "SET_USER",
            payload = null
          )
          dispatch(action)
        }
        () => ()
      }, js.Array())
      
      // Login function
      val login = js.Function2 { (email: String, password: String) =>
        dispatch(js.Dynamic.literal(`type` = "SET_LOADING"))
        
        AuthService.login(email, password).foreach { response =>
          if (response.success) {
            dispatch(js.Dynamic.literal(
              `type` = "LOGIN_SUCCESS",
              payload = response.user.get.toJS
            ))
          } else {
            dispatch(js.Dynamic.literal(
              `type` = "LOGIN_FAILURE",
              payload = response.message
            ))
          }
        }
      }
      
      // Logout function
      val logout = js.Function0 { () =>
        AuthService.logout()
        dispatch(js.Dynamic.literal(`type` = "LOGOUT"))
      }
      
      // Create context value
      val value = js.Dynamic.literal(
        state = state,
        login = login,
        logout = logout
      )
      
      // Render provider with value and children
      val providerProps = js.Dynamic.literal(value = value).asInstanceOf[js.Object]
      React.createElement(authContext.Provider, providerProps, children)
    }
  }
  
  // Hook to use auth context
  def useAuth(): js.Dynamic = {
    val context = React.useContext(authContext)
    if (context == null) {
      throw new RuntimeException("useAuth must be used within an AuthProvider")
    }
    context
  }
}

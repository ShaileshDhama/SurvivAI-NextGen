package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import org.scalajs.dom
import survivai.bindings.ReactBindings.*
import survivai.contexts.AuthContext

object Login {
  def render(): Element = {
    FC {
      // Get auth context
      val auth = AuthContext.useAuth()
      
      // Form state hooks
      val (email, setEmail) = useState("")
      val (password, setPassword) = useState("")
      val (isLoading, setIsLoading) = useState(false)
      val (error, setError) = useState[Option[String]](None)
      
      // Check if already authenticated and redirect if needed
      useEffect(() => {
        if (auth.state.isAuthenticated.asInstanceOf[Boolean]) {
          dom.window.location.href = "#/"
        }
      }, js.Array(auth.state.isAuthenticated))
      
      // Handle login form submission
      val handleSubmit = (event: ReactEventFrom[dom.html.Form]) => {
        event.preventDefault()
        
        // Validate form
        if (email.trim.isEmpty || password.isEmpty) {
          setError(Some("Please enter both email and password"))
        } else {
          setIsLoading(true)
          setError(None)
          
          val loginFn = auth.login.asInstanceOf[js.Function2[String, String, Unit]]
          loginFn(email, password)
          
          // Note: we don't need to handle redirect here as the AuthContext will
          // update its state which will trigger the useEffect above
        }
      }
      
      // Use the Scala.js React API to create the login form
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = "flex flex-col items-center justify-center min-h-screen p-6 bg-gray-50"
        ).asInstanceOf[js.Object],
        
        // Card container
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "w-full max-w-md p-8 space-y-8 bg-white rounded-lg shadow-md"
          ).asInstanceOf[js.Object],
          
          // Logo
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "text-4xl font-bold text-center text-blue-600"
            ).asInstanceOf[js.Object],
            "SurvivAI"
          ),
          
          // Title
          React.createElement(
            "h1",
            js.Dynamic.literal(
              className = "mt-6 text-2xl font-semibold text-center text-gray-900"
            ).asInstanceOf[js.Object],
            "Log in to your account"
          ),
          
          // Form
          React.createElement(
            "form",
            js.Dynamic.literal(
              className = "mt-8 space-y-6",
              onSubmit = handleSubmit
            ).asInstanceOf[js.Object],
            
            // Email field
            React.createElement(
              "div",
              js.Dynamic.literal().asInstanceOf[js.Object],
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "email",
                  className = "block text-sm font-medium text-gray-700"
                ).asInstanceOf[js.Object],
                "Email address"
              ),
              
              React.createElement(
                "input",
                js.Dynamic.literal(
                  id = "email",
                  name = "email",
                  type = "email",
                  required = true,
                  className = "block w-full px-3 py-2 mt-1 text-gray-900 placeholder-gray-400 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm",
                  autoComplete = "email",
                  value = email,
                  onChange = (e: ReactEventFrom[dom.html.Input]) => setEmail(e.target.value),
                  disabled = isLoading
                ).asInstanceOf[js.Object]
              )
            ),
            
            // Password field
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "mt-6"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "password",
                  className = "block text-sm font-medium text-gray-700"
                ).asInstanceOf[js.Object],
                "Password"
              ),
              
              React.createElement(
                "input",
                js.Dynamic.literal(
                  id = "password",
                  name = "password",
                  type = "password",
                  required = true,
                  className = "block w-full px-3 py-2 mt-1 text-gray-900 placeholder-gray-400 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm",
                  autoComplete = "current-password",
                  value = password,
                  onChange = (e: ReactEventFrom[dom.html.Input]) => setPassword(e.target.value),
                  disabled = isLoading
                ).asInstanceOf[js.Object]
              )
            ),
            
            // Login button
            React.createElement(
              "button",
              js.Dynamic.literal(
                type = "submit",
                className = "flex justify-center w-full px-4 py-2 mt-6 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400 disabled:cursor-not-allowed",
                disabled = isLoading
              ).asInstanceOf[js.Object],
              if (isLoading) "Logging in..." else "Sign in"
            ),
            
            // Error message
            error.map { errorMessage =>
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "mt-4 text-sm text-center text-red-600"
                ).asInstanceOf[js.Object],
                errorMessage
              )
            }.getOrElse(null),
            
            // Forgot password link
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "flex items-center justify-center mt-6"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "a",
                js.Dynamic.literal(
                  href = "#/forgot-password",
                  className = "text-sm font-medium text-blue-600 hover:text-blue-500"
                ).asInstanceOf[js.Object],
                "Forgot your password?"
              )
            )
          )
        )
      )
    }
  }
}

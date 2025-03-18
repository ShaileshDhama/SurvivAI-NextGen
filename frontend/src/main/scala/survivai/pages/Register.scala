package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import org.scalajs.dom
import survivai.bindings.ReactBindings.*
import survivai.contexts.AuthContext

object Register {
  def render(): Element = {
    FC {
      // State hooks
      val (name, setName) = useState("")
      val (email, setEmail) = useState("")
      val (password, setPassword) = useState("")
      val (confirmPassword, setConfirmPassword) = useState("")
      val (error, setError) = useState[Option[String]](None)
      val (isLoading, setIsLoading) = useState(false)
      
      // Handle form submission
      val handleSubmit = (event: ReactEventFrom[dom.html.Form]) => {
        event.preventDefault()
        
        // Validate form
        if (name.trim.isEmpty || email.trim.isEmpty || password.isEmpty) {
          setError(Some("Please fill out all required fields"))
          return
        }
        
        if (password != confirmPassword) {
          setError(Some("Passwords do not match"))
          return
        }
        
        setIsLoading(true)
        setError(None)
        
        // In a real application, we would call the AuthService to register
        // For now, just redirect to login page with success message
        js.timers.setTimeout(1000) {
          setIsLoading(false)
          // Navigate to login page
          dom.window.location.href = "#/login"
        }
      }
      
      // Render the registration form
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
          
          // Logo and title
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "text-4xl font-bold text-center text-blue-600"
            ).asInstanceOf[js.Object],
            "SurvivAI"
          ),
          
          React.createElement(
            "h1",
            js.Dynamic.literal(
              className = "mt-6 text-2xl font-semibold text-center text-gray-900"
            ).asInstanceOf[js.Object],
            "Create your account"
          ),
          
          // Registration form
          React.createElement(
            "form",
            js.Dynamic.literal(
              className = "mt-8 space-y-6",
              onSubmit = handleSubmit
            ).asInstanceOf[js.Object],
            
            // Full name field
            React.createElement(
              "div",
              js.Dynamic.literal().asInstanceOf[js.Object],
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "name",
                  className = "block text-sm font-medium text-gray-700"
                ).asInstanceOf[js.Object],
                "Full name"
              ),
              
              React.createElement(
                "input",
                js.Dynamic.literal(
                  id = "name",
                  name = "name",
                  type = "text",
                  required = true,
                  className = "block w-full px-3 py-2 mt-1 text-gray-900 placeholder-gray-400 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm",
                  value = name,
                  onChange = (e: ReactEventFrom[dom.html.Input]) => setName(e.target.value),
                  disabled = isLoading
                ).asInstanceOf[js.Object]
              )
            ),
            
            // Email field
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "mt-6"
              ).asInstanceOf[js.Object],
              
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
                  value = password,
                  onChange = (e: ReactEventFrom[dom.html.Input]) => setPassword(e.target.value),
                  disabled = isLoading
                ).asInstanceOf[js.Object]
              )
            ),
            
            // Confirm password field
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "mt-6"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "confirmPassword",
                  className = "block text-sm font-medium text-gray-700"
                ).asInstanceOf[js.Object],
                "Confirm password"
              ),
              
              React.createElement(
                "input",
                js.Dynamic.literal(
                  id = "confirmPassword",
                  name = "confirmPassword",
                  type = "password",
                  required = true,
                  className = "block w-full px-3 py-2 mt-1 text-gray-900 placeholder-gray-400 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm",
                  value = confirmPassword,
                  onChange = (e: ReactEventFrom[dom.html.Input]) => setConfirmPassword(e.target.value),
                  disabled = isLoading
                ).asInstanceOf[js.Object]
              )
            ),
            
            // Error message (if any)
            error.map { errorMessage =>
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "mt-4 text-sm text-center text-red-600"
                ).asInstanceOf[js.Object],
                errorMessage
              )
            }.getOrElse(null),
            
            // Register button
            React.createElement(
              "button",
              js.Dynamic.literal(
                type = "submit",
                className = "flex justify-center w-full px-4 py-2 mt-6 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400 disabled:cursor-not-allowed",
                disabled = isLoading
              ).asInstanceOf[js.Object],
              if (isLoading) "Creating account..." else "Create account"
            ),
            
            // Sign in link
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "flex items-center justify-center mt-6"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "span",
                js.Dynamic.literal(
                  className = "text-sm text-gray-600"
                ).asInstanceOf[js.Object],
                "Already have an account? "
              ),
              
              React.createElement(
                "a",
                js.Dynamic.literal(
                  href = "#/login",
                  className = "text-sm font-medium text-blue-600 hover:text-blue-500 ml-1"
                ).asInstanceOf[js.Object],
                "Sign in"
              )
            )
          )
        )
      )
    }
  }
}

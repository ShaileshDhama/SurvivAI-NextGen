package survivai.components.layout

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*
import survivai.components.*
import survivai.contexts.{AuthContext, LayoutContext}

object Navbar {
  def render(): Element = {
    FC {
      // Get context values
      val auth = AuthContext.useAuth()
      val layout = LayoutContext.useLayout()
      
      // Extract user information
      val isAuthenticated = auth.state.isAuthenticated.asInstanceOf[Boolean]
      val user = if (isAuthenticated) auth.state.user.asInstanceOf[js.Dynamic] else null
      val title = layout.state.title.asInstanceOf[String]
      
      // Handle logout
      val handleLogout = () => {
        val logoutFn = auth.logout.asInstanceOf[js.Function0[Unit]]
        logoutFn()
      }
      
      val navbarProps = js.Dynamic.literal(
        className = "bg-white shadow-md h-16 flex items-center justify-between px-6"
      )
      
      React.createElement(
        "div",
        navbarProps.asInstanceOf[js.Object],
        
        // Left side - page title
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "text-xl font-semibold text-gray-700"
          ).asInstanceOf[js.Object],
          title
        ),
        
        // Right side - user menu
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "flex items-center space-x-4"
          ).asInstanceOf[js.Object],
          
          // Notifications
          React.createElement(
            "button",
            js.Dynamic.literal(
              className = "p-2 rounded-full hover:bg-gray-100"
            ).asInstanceOf[js.Object],
            // Notification bell icon placeholder
            "🔔"
          ),
          
          // User profile or login button
          if (isAuthenticated && user != null) {
            // User profile dropdown
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "flex items-center space-x-2 cursor-pointer"
              ).asInstanceOf[js.Object],
              
              // Avatar
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white"
                ).asInstanceOf[js.Object],
                user.username.asInstanceOf[String].substring(0, 1).toUpperCase
              ),
              
              // Username
              React.createElement(
                "span",
                js.Dynamic.literal(
                  className = "text-sm font-medium text-gray-700"
                ).asInstanceOf[js.Object],
                user.username.asInstanceOf[String]
              ),
              
              // Logout button
              React.createElement(
                "button",
                js.Dynamic.literal(
                  className = "ml-2 text-sm text-gray-500 hover:text-gray-700",
                  onClick = handleLogout
                ).asInstanceOf[js.Object],
                "Logout"
              )
            )
          } else {
            // Login button
            React.createElement(
              "a",
              js.Dynamic.literal(
                href = "#/login",
                className = "text-sm font-medium text-blue-600 hover:text-blue-500"
              ).asInstanceOf[js.Object],
              "Login"
            )
          }
        )
      )
    }
  }
}

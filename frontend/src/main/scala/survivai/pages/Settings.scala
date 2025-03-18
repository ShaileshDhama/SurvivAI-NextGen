package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Success, Failure}
import survivai.bindings.ReactBindings.*
import survivai.contexts.LayoutContext

object Settings {
  def render(): Element = {
    FC {
      // Set the page title
      val layoutContext = LayoutContext.useLayout()
      layoutContext.setTitle("Settings")
      
      // State for user preferences
      val themeState = React.useState[String]("light")
      val theme = themeState(0).asInstanceOf[String]
      val setTheme = themeState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val notificationsState = React.useState[Boolean](true)
      val notifications = notificationsState(0).asInstanceOf[Boolean]
      val setNotifications = notificationsState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      val dataDecimalsState = React.useState[Int](2)
      val dataDecimals = dataDecimalsState(0).asInstanceOf[Int]
      val setDataDecimals = dataDecimalsState(1).asInstanceOf[js.Function1[Int, Unit]]
      
      val messageState = React.useState[Option[(String, Boolean)]](None) // (message, isError)
      val message = messageState(0).asInstanceOf[Option[(String, Boolean)]]
      val setMessage = messageState(1).asInstanceOf[js.Function1[Option[(String, Boolean)], Unit]]
      
      // Save settings
      val saveSettings = js.Function1 { (event: js.Dynamic) =>
        event.preventDefault()
        
        // Save to localStorage for now
        // In a real app, you would save to user profile in backend
        try {
          val localStorage = js.Dynamic.global.window.localStorage
          localStorage.setItem("survivai-theme", theme)
          localStorage.setItem("survivai-notifications", notifications.toString())
          localStorage.setItem("survivai-data-decimals", dataDecimals.toString())
          
          setMessage(Some(("Settings saved successfully.", false)))
          
          // Auto-hide message after 3 seconds
          js.timers.setTimeout(3000) {
            setMessage(None)
          }
        } catch {
          case e: Throwable =>
            setMessage(Some((s"Error saving settings: ${e.getMessage}", true)))
        }
      }
      
      // Load settings from localStorage on mount
      React.useEffect(() => {
        try {
          val localStorage = js.Dynamic.global.window.localStorage
          
          // Load theme
          val savedTheme = localStorage.getItem("survivai-theme").asInstanceOf[String]
          if (savedTheme != null && savedTheme.nonEmpty) {
            setTheme(savedTheme)
          }
          
          // Load notifications
          val savedNotifications = localStorage.getItem("survivai-notifications").asInstanceOf[String]
          if (savedNotifications != null && savedNotifications.nonEmpty) {
            setNotifications(savedNotifications.toBoolean)
          }
          
          // Load data decimals
          val savedDecimals = localStorage.getItem("survivai-data-decimals").asInstanceOf[String]
          if (savedDecimals != null && savedDecimals.nonEmpty) {
            setDataDecimals(savedDecimals.toInt)
          }
        } catch {
          case _: Throwable => 
            // Ignore errors, use defaults
        }
        
        () => ()
      }, js.Array())
      
      // Handle theme change
      val handleThemeChange = js.Function1 { (event: js.Dynamic) =>
        setTheme(event.target.value.asInstanceOf[String])
      }
      
      // Handle notifications change
      val handleNotificationsChange = js.Function1 { (event: js.Dynamic) =>
        setNotifications(event.target.checked.asInstanceOf[Boolean])
      }
      
      // Handle data decimals change
      val handleDataDecimalsChange = js.Function1 { (event: js.Dynamic) =>
        setDataDecimals(event.target.value.asInstanceOf[Int])
      }
      
      // Render settings page
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = "space-y-6"
        ).asInstanceOf[js.Object],
        
        // Page header
        React.createElement(
          "div",
          null,
          
          React.createElement(
            "h1",
            js.Dynamic.literal(
              className = "text-2xl font-bold"
            ).asInstanceOf[js.Object],
            "Settings"
          ),
          
          React.createElement(
            "p",
            js.Dynamic.literal(
              className = "text-gray-600 mt-1"
            ).asInstanceOf[js.Object],
            "Customize your SurvivAI experience"
          )
        ),
        
        // Status message
        message.map { case (msg, isError) =>
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = if (isError) "bg-red-50 border border-red-200 text-red-800 rounded-md p-4" 
                         else "bg-green-50 border border-green-200 text-green-800 rounded-md p-4"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "p",
              null,
              msg
            )
          )
        }.getOrElse(null),
        
        // Settings form
        React.createElement(
          "form",
          js.Dynamic.literal(
            onSubmit = saveSettings,
            className = "bg-white shadow rounded-lg p-6 space-y-6"
          ).asInstanceOf[js.Object],
          
          // Appearance section
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "space-y-4"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "h3",
              js.Dynamic.literal(
                className = "text-lg font-medium"
              ).asInstanceOf[js.Object],
              "Appearance"
            ),
            
            // Theme setting
            React.createElement(
              "div",
              null,
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "theme",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "Theme"
              ),
              
              React.createElement(
                "select",
                js.Dynamic.literal(
                  id = "theme",
                  name = "theme",
                  value = theme,
                  onChange = handleThemeChange,
                  className = "mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = "light"
                  ).asInstanceOf[js.Object],
                  "Light"
                ),
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = "dark"
                  ).asInstanceOf[js.Object],
                  "Dark"
                ),
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = "system"
                  ).asInstanceOf[js.Object],
                  "System Preference"
                )
              )
            )
          ),
          
          // Notifications section
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "space-y-4 pt-6 border-t border-gray-200"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "h3",
              js.Dynamic.literal(
                className = "text-lg font-medium"
              ).asInstanceOf[js.Object],
              "Notifications"
            ),
            
            // Enable notifications
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "flex items-start"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "flex items-center h-5"
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "input",
                  js.Dynamic.literal(
                    id = "notifications",
                    name = "notifications",
                    type = "checkbox",
                    checked = notifications,
                    onChange = handleNotificationsChange,
                    className = "focus:ring-blue-500 h-4 w-4 text-blue-600 border-gray-300 rounded"
                  ).asInstanceOf[js.Object]
                )
              ),
              
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "ml-3 text-sm"
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "label",
                  js.Dynamic.literal(
                    htmlFor = "notifications",
                    className = "font-medium text-gray-700"
                  ).asInstanceOf[js.Object],
                  "Enable Notifications"
                ),
                
                React.createElement(
                  "p",
                  js.Dynamic.literal(
                    className = "text-gray-500"
                  ).asInstanceOf[js.Object],
                  "Receive notifications about completed analyses and system updates."
                )
              )
            )
          ),
          
          // Data Display section
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "space-y-4 pt-6 border-t border-gray-200"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "h3",
              js.Dynamic.literal(
                className = "text-lg font-medium"
              ).asInstanceOf[js.Object],
              "Data Display"
            ),
            
            // Decimal places
            React.createElement(
              "div",
              null,
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "data-decimals",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "Decimal Places"
              ),
              
              React.createElement(
                "select",
                js.Dynamic.literal(
                  id = "data-decimals",
                  name = "data-decimals",
                  value = dataDecimals,
                  onChange = handleDataDecimalsChange,
                  className = "mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                ).asInstanceOf[js.Object],
                
                (0 to 6).map { num =>
                  React.createElement(
                    "option",
                    js.Dynamic.literal(
                      key = num,
                      value = num
                    ).asInstanceOf[js.Object],
                    num.toString
                  )
                }: _*
              ),
              
              React.createElement(
                "p",
                js.Dynamic.literal(
                  className = "mt-1 text-sm text-gray-500"
                ).asInstanceOf[js.Object],
                "Number of decimal places to display in numerical data."
              )
            )
          ),
          
          // Submit button
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "pt-6 border-t border-gray-200"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "button",
              js.Dynamic.literal(
                type = "submit",
                className = "w-full bg-blue-600 text-white py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              ).asInstanceOf[js.Object],
              "Save Settings"
            )
          )
        )
      )
    }
  }
}

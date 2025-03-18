package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Success, Failure}
import survivai.bindings.ReactBindings.*
import survivai.services.AuthService
import survivai.contexts.{AuthContext, LayoutContext}

object UserProfile {
  def render(): Element = {
    FC {
      // Set the page title
      val layoutContext = LayoutContext.useLayout()
      layoutContext.setTitle("User Profile")
      
      // Get authentication context
      val authContext = AuthContext.useAuth()
      val user = authContext.state.user.asInstanceOf[js.Dynamic]
      
      // Form states
      val nameState = React.useState[String](if (user != null) user.name.asInstanceOf[String] else "")
      val name = nameState(0).asInstanceOf[String]
      val setName = nameState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val emailState = React.useState[String](if (user != null) user.email.asInstanceOf[String] else "")
      val email = emailState(0).asInstanceOf[String]
      val setEmail = emailState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val currentPasswordState = React.useState[String]("")
      val currentPassword = currentPasswordState(0).asInstanceOf[String]
      val setCurrentPassword = currentPasswordState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val newPasswordState = React.useState[String]("")
      val newPassword = newPasswordState(0).asInstanceOf[String]
      val setNewPassword = newPasswordState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val confirmPasswordState = React.useState[String]("")
      val confirmPassword = confirmPasswordState(0).asInstanceOf[String]
      val setConfirmPassword = confirmPasswordState(1).asInstanceOf[js.Function1[String, Unit]]
      
      // Status states
      val savingState = React.useState[Boolean](false)
      val saving = savingState(0).asInstanceOf[Boolean]
      val setSaving = savingState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      val messageState = React.useState[Option[(String, Boolean)]](None) // (message, isError)
      val message = messageState(0).asInstanceOf[Option[(String, Boolean)]]
      val setMessage = messageState(1).asInstanceOf[js.Function1[Option[(String, Boolean)], Unit]]
      
      // Reset form when user changes
      React.useEffect(() => {
        if (user != null) {
          setName(user.name.asInstanceOf[String])
          setEmail(user.email.asInstanceOf[String])
        }
        () => ()
      }, js.Array(user))
      
      // Handle form submission for profile update
      val handleProfileUpdate = js.Function1 { (event: js.Dynamic) =>
        event.preventDefault()
        setSaving(true)
        setMessage(None)
        
        // Call API to update profile
        AuthService.updateProfile(name, email).onComplete {
          case Success(response) =>
            setSaving(false)
            if (response.success) {
              setMessage(Some(("Profile updated successfully.", false)))
              // Update user in context
              authContext.login(email, currentPassword)
            } else {
              setMessage(Some((response.message, true)))
            }
          case Failure(exception) =>
            setSaving(false)
            setMessage(Some((s"Error: ${exception.getMessage}", true)))
        }
      }
      
      // Handle password change
      val handlePasswordChange = js.Function1 { (event: js.Dynamic) =>
        event.preventDefault()
        
        // Validate passwords
        if (newPassword != confirmPassword) {
          setMessage(Some(("New passwords do not match.", true)))
          return
        }
        
        if (newPassword.length < 8) {
          setMessage(Some(("Password must be at least 8 characters long.", true)))
          return
        }
        
        setSaving(true)
        setMessage(None)
        
        // Call API to change password
        AuthService.changePassword(currentPassword, newPassword).onComplete {
          case Success(response) =>
            setSaving(false)
            if (response.success) {
              setMessage(Some(("Password changed successfully.", false)))
              setCurrentPassword("")
              setNewPassword("")
              setConfirmPassword("")
            } else {
              setMessage(Some((response.message, true)))
            }
          case Failure(exception) =>
            setSaving(false)
            setMessage(Some((s"Error: ${exception.getMessage}", true)))
        }
      }
      
      // Helper function for input change handlers
      def createChangeHandler(setter: js.Function1[String, Unit]): js.Function1[js.Dynamic, Unit] = {
        js.Function1 { (event: js.Dynamic) =>
          setter(event.target.value.asInstanceOf[String])
        }
      }
      
      // Render profile page
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
            "User Profile"
          ),
          
          React.createElement(
            "p",
            js.Dynamic.literal(
              className = "text-gray-600 mt-1"
            ).asInstanceOf[js.Object],
            "Manage your account information and password"
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
        
        // Profile section
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-white shadow rounded-lg p-6"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "h2",
            js.Dynamic.literal(
              className = "text-lg font-medium mb-6"
            ).asInstanceOf[js.Object],
            "Profile Information"
          ),
          
          React.createElement(
            "form",
            js.Dynamic.literal(
              onSubmit = handleProfileUpdate,
              className = "space-y-4"
            ).asInstanceOf[js.Object],
            
            // Name field
            React.createElement(
              "div",
              null,
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "name",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "Name"
              ),
              
              React.createElement(
                "input",
                js.Dynamic.literal(
                  id = "name",
                  name = "name",
                  type = "text",
                  value = name,
                  onChange = createChangeHandler(setName),
                  required = true,
                  className = "appearance-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm"
                ).asInstanceOf[js.Object]
              )
            ),
            
            // Email field
            React.createElement(
              "div",
              null,
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "email",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "Email Address"
              ),
              
              React.createElement(
                "input",
                js.Dynamic.literal(
                  id = "email",
                  name = "email",
                  type = "email",
                  value = email,
                  onChange = createChangeHandler(setEmail),
                  required = true,
                  className = "appearance-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm"
                ).asInstanceOf[js.Object]
              )
            ),
            
            // Submit button
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "pt-2"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "button",
                js.Dynamic.literal(
                  type = "submit",
                  disabled = saving,
                  className = s"w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${if (saving) "opacity-70 cursor-not-allowed" else ""}"
                ).asInstanceOf[js.Object],
                if (saving) "Saving..." else "Save Profile"
              )
            )
          )
        ),
        
        // Password section
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-white shadow rounded-lg p-6"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "h2",
            js.Dynamic.literal(
              className = "text-lg font-medium mb-6"
            ).asInstanceOf[js.Object],
            "Change Password"
          ),
          
          React.createElement(
            "form",
            js.Dynamic.literal(
              onSubmit = handlePasswordChange,
              className = "space-y-4"
            ).asInstanceOf[js.Object],
            
            // Current password field
            React.createElement(
              "div",
              null,
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "current-password",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "Current Password"
              ),
              
              React.createElement(
                "input",
                js.Dynamic.literal(
                  id = "current-password",
                  name = "current-password",
                  type = "password",
                  value = currentPassword,
                  onChange = createChangeHandler(setCurrentPassword),
                  required = true,
                  className = "appearance-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm"
                ).asInstanceOf[js.Object]
              )
            ),
            
            // New password field
            React.createElement(
              "div",
              null,
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "new-password",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "New Password"
              ),
              
              React.createElement(
                "input",
                js.Dynamic.literal(
                  id = "new-password",
                  name = "new-password",
                  type = "password",
                  value = newPassword,
                  onChange = createChangeHandler(setNewPassword),
                  required = true,
                  minLength = 8,
                  className = "appearance-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm"
                ).asInstanceOf[js.Object]
              ),
              
              React.createElement(
                "p",
                js.Dynamic.literal(
                  className = "mt-1 text-xs text-gray-500"
                ).asInstanceOf[js.Object],
                "Password must be at least 8 characters long"
              )
            ),
            
            // Confirm password field
            React.createElement(
              "div",
              null,
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "confirm-password",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "Confirm New Password"
              ),
              
              React.createElement(
                "input",
                js.Dynamic.literal(
                  id = "confirm-password",
                  name = "confirm-password",
                  type = "password",
                  value = confirmPassword,
                  onChange = createChangeHandler(setConfirmPassword),
                  required = true,
                  className = "appearance-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm"
                ).asInstanceOf[js.Object]
              )
            ),
            
            // Submit button
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "pt-2"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "button",
                js.Dynamic.literal(
                  type = "submit",
                  disabled = saving,
                  className = s"w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${if (saving) "opacity-70 cursor-not-allowed" else ""}"
                ).asInstanceOf[js.Object],
                if (saving) "Changing Password..." else "Change Password"
              )
            )
          )
        )
      )
    }
  }
}

package survivai.routes

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*
import survivai.components.*
import survivai.components.layout.Layout
import survivai.pages.*
import survivai.contexts.AuthContext

object Routes {
  def render(): Element = {
    FC {
      // Get auth context
      val authContext = AuthContext.useAuth()
      val isAuthenticated = authContext.isAuthenticated
      
      // Get current hash location
      val locationState = React.useState[String]("/")
      val location = locationState(0).asInstanceOf[String]
      val setLocation = locationState(1).asInstanceOf[js.Function1[String, Unit]]
      
      // Handle hash change
      React.useEffect(() => {
        // Initialize from current hash
        val currentHash = js.Dynamic.global.window.location.hash.asInstanceOf[String]
        val path = if (currentHash.startsWith("#")) currentHash.substring(1) else "/"
        setLocation(path)
        
        // Listen for hash changes
        val handleHashChange = () => {
          val hash = js.Dynamic.global.window.location.hash.asInstanceOf[String]
          val newPath = if (hash.startsWith("#")) hash.substring(1) else "/"
          setLocation(newPath)
        }
        
        js.Dynamic.global.window.addEventListener("hashchange", handleHashChange)
        
        // Cleanup
        () => {
          js.Dynamic.global.window.removeEventListener("hashchange", handleHashChange)
        }
      }, js.Array())
      
      // Render appropriate page based on location
      val content = if (!isAuthenticated) {
        // Not authenticated - show login
        if (location == "/register") {
          Register.render()
        } else {
          Login.render()
        }
      } else {
        // Authenticated - show appropriate page
        location match {
          // Dashboard
          case "/" => Dashboard.render()
          
          // Datasets
          case "/datasets" => DatasetManagement.render()
          case "/datasets/new" => UploadDataset.render()
          case path if path.startsWith("/datasets/") => DatasetDetail.render(path.split("/").last)
          
          // Analyses
          case "/analyses" => AnalysisList.render()
          case "/analyses/new" => NewAnalysis.render()
          case path if path.startsWith("/analyses/") => AnalysisDetail.render(path.split("/").last)
          
          // Reports
          case "/reports" => Reports.render()
          case path if path.startsWith("/reports/") => Reports.render() // We handle individual reports within the Reports component
          
          // Visualizations
          case "/visualizations" => VisualizationList.render()
          case "/visualizations/new" => CreateVisualization.render()
          case path if path.startsWith("/visualizations/") => VisualizationDetail.render(path.split("/").last)
          
          // Profile
          case "/profile" => UserProfile.render()
          
          // Settings
          case "/settings" => Settings.render()
          
          // 404 - Default to dashboard if not found
          case _ => Dashboard.render()
        }
      }
      
      // Return the content wrapped in a fragment
      content
    }
  }
  
  // Place holder components (stubs) for remaining pages that haven't been fully migrated yet
  object DatasetDetail {
    def render(id: String): Element = FC {
      Layout.render(
        Layout.Props(
          children = React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-white p-6 rounded-lg shadow-md"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "h2",
              js.Dynamic.literal(
                className = "text-xl font-semibold mb-4"
              ).asInstanceOf[js.Object],
              s"Dataset Details (ID: $id)"
            ),
            
            React.createElement(
              "p",
              js.Dynamic.literal(
                className = "text-gray-600"
              ).asInstanceOf[js.Object],
              "This is a placeholder for the dataset detail page. This component will be fully implemented in Scala.js soon."
            )
          ),
          title = "Dataset Details",
          subtitle = s"Viewing details for dataset $id"
        )
      )
    }
  }
}

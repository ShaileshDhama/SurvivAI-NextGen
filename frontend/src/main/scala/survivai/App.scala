package survivai

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import org.scalajs.dom
import survivai.bindings.ReactBindings.*
import survivai.pages.*
import survivai.contexts.*

object App {
  def render(): Element = {
    FC {
      // Get auth context
      val authContext = AuthContext.useAuth()
      val isAuthenticated = authContext.isAuthenticated
      
      // State for route handling
      val (currentRoute, setCurrentRoute) = useState(() => getRouteFromHash())
      
      // Listen to hash changes for routing
      useEffect(() => {
        val handleHashChange = () => {
          setCurrentRoute(getRouteFromHash())
        }
        
        // Add event listener
        dom.window.addEventListener("hashchange", handleHashChange)
        
        // Clean up function
        () => {
          dom.window.removeEventListener("hashchange", handleHashChange)
        }
      }, js.Array())
      
      // Parse route from URL hash
      def getRouteFromHash(): String = {
        val hash = dom.window.location.hash
        if (hash.isEmpty || hash == "#") {
          "#/"
        } else {
          hash
        }
      }
      
      // Function to render the appropriate component based on route
      def renderRoute(): Element = {
        if (!isAuthenticated) {
          // Not authenticated - show login
          if (currentRoute == "#/register") {
            // We would render Register page here if implemented
            Login.render()
          } else {
            Login.render()
          }
        } else {
          // Authenticated - show appropriate page
          currentRoute match {
            case "#/" | "#/dashboard" => Dashboard.render()
            
            // Datasets
            case "#/datasets" => Datasets.render()
            case "#/datasets/new" => UploadDataset.render() // Assuming this is implemented
            case r if r.startsWith("#/datasets/") => 
              val id = r.substring("#/datasets/".length)
              DatasetDetail.render(id) // Placeholder until full implementation
            
            // Analyses
            case "#/analyses" => Analyses.render()
            case "#/analyses/new" => NewAnalysis.render()
            case r if r.startsWith("#/analyses/") => 
              val id = r.substring("#/analyses/".length)
              AnalysisDetail.render(id) // Placeholder until full implementation
            
            // Models
            case "#/models" => Models.render()
            case "#/models/new" => NewModel.render()
            case r if r.startsWith("#/models/") => 
              val id = r.substring("#/models/".length)
              ModelDetail.render(id) // Placeholder until full implementation
              
            // Visualizations
            case "#/visualizations" => Visualizations.render()
            case "#/visualizations/new" => NewVisualization.render()
            case r if r.startsWith("#/visualizations/") => 
              val id = r.substring("#/visualizations/".length)
              VisualizationDetail.render(id) // Placeholder until full implementation
            
            // Reports
            case "#/reports" => Reports.render()
            case r if r.startsWith("#/reports/") => 
              Reports.render() // We handle individual reports within the Reports component
            
            // User profile and settings
            case "#/profile" => Dashboard.render() // Placeholder until UserProfile is implemented
            case "#/settings" => Dashboard.render() // Placeholder until Settings is implemented
            
            // Default to dashboard if not found
            case _ => 
              dom.window.location.hash = "#/dashboard"
              null
          }
        }
      }
      
      // Placeholder components for pages not yet fully implemented
      object DatasetDetail {
        def render(id: String): Element = {
          import survivai.components.layout.Layout
          
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
                  "This page is currently being migrated to Scala.js."
                )
              ),
              title = "Dataset Details",
              subtitle = s"Viewing details for dataset $id"
            )
          )
        }
      }
      
      object AnalysisDetail {
        def render(id: String): Element = {
          import survivai.components.layout.Layout
          
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
                  s"Analysis Details (ID: $id)"
                ),
                
                React.createElement(
                  "p",
                  js.Dynamic.literal(
                    className = "text-gray-600"
                  ).asInstanceOf[js.Object],
                  "This page is currently being migrated to Scala.js."
                )
              ),
              title = "Analysis Details",
              subtitle = s"Viewing details for analysis $id"
            )
          )
        }
      }
      
      object ModelDetail {
        def render(id: String): Element = {
          import survivai.components.layout.Layout
          
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
                  s"Model Details (ID: $id)"
                ),
                
                React.createElement(
                  "p",
                  js.Dynamic.literal(
                    className = "text-gray-600"
                  ).asInstanceOf[js.Object],
                  "This page is currently being migrated to Scala.js."
                )
              ),
              title = "Model Details",
              subtitle = s"Viewing details for model $id"
            )
          )
        }
      }
      
      object VisualizationDetail {
        def render(id: String): Element = {
          import survivai.components.layout.Layout
          
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
                  s"Visualization Details (ID: $id)"
                ),
                
                React.createElement(
                  "p",
                  js.Dynamic.literal(
                    className = "text-gray-600"
                  ).asInstanceOf[js.Object],
                  "This page is currently being migrated to Scala.js."
                )
              ),
              title = "Visualization Details",
              subtitle = s"Viewing details for visualization $id"
            )
          )
        }
      }
      
      // Wrap the app with context providers
      React.createElement(
        AuthContext.Provider,
        null,
        React.createElement(
          AnalysisContext.Provider,
          null,
          React.createElement(
            DatasetContext.Provider,
            null,
            renderRoute()
          )
        )
      )
    }
  }
}

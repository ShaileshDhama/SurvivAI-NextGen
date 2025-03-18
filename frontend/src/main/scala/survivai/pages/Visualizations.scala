package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import survivai.bindings.ReactBindings.*
import survivai.components.*
import survivai.components.layout.Layout
import survivai.services.VisualizationService
import survivai.models.Visualization
import java.util.Date
import org.scalajs.dom

object Visualizations {
  def render(): Element = {
    FC {
      // State hooks
      val (visualizations, setVisualizations) = useState(js.Array[Visualization.Visualization]())
      val (isLoading, setIsLoading) = useState(true)
      val (searchTerm, setSearchTerm) = useState("")
      val (filter, setFilter) = useState("all")
      val (error, setError) = useState[Option[String]](None)
      
      // Fetch visualizations on component mount
      useEffect(() => {
        fetchVisualizations()
        () => ()
      }, js.Array())
      
      // Fetch visualizations function
      def fetchVisualizations(): Unit = {
        setIsLoading(true)
        setError(None)
        
        VisualizationService.getVisualizations(None).foreach { result =>
          setVisualizations(js.Array(result: _*))
          setIsLoading(false)
        } recover { case ex => 
          console.error("Failed to fetch visualizations:", ex.getMessage)
          setError(Some(s"Failed to load visualizations: ${ex.getMessage}"))
          setIsLoading(false)
        }
      }
      
      // Filter visualizations based on search term and filter
      val filteredVisualizations = visualizations.filter { viz =>
        val matchesSearch = searchTerm.isEmpty || 
          viz.title.toLowerCase.contains(searchTerm.toLowerCase) || 
          viz.description.exists(_.toLowerCase.contains(searchTerm.toLowerCase))
        
        val matchesFilter = filter match {
          case "all" => true
          case "kaplan-meier" => viz.vizType == Visualization.KaplanMeier
          case "hazard-ratio" => viz.vizType == Visualization.HazardRatio
          case "feature-importance" => viz.vizType == Visualization.FeatureImportance
          case _ => true
        }
        
        matchesSearch && matchesFilter
      }
      
      // Handle visualization selection
      val handleSelectVisualization = (vizId: String) => {
        dom.window.location.href = s"#/visualizations/${vizId}"
      }
      
      // Handle create new visualization
      val handleCreateVisualization = () => {
        dom.window.location.href = "#/visualizations/new"
      }
      
      // Handle delete visualization
      val handleDeleteVisualization = (vizId: String) => {
        if (dom.window.confirm("Are you sure you want to delete this visualization? This action cannot be undone.")) {
          VisualizationService.deleteVisualization(vizId).foreach { success =>
            if (success) {
              fetchVisualizations()
            } else {
              setError(Some("Failed to delete visualization"))
            }
          }
        }
      }
      
      // Handle search term change
      val handleSearchChange = (e: ReactEventFrom[dom.html.Input]) => {
        setSearchTerm(e.target.value)
      }
      
      // Handle filter change
      val handleFilterChange = (e: ReactEventFrom[dom.html.Select]) => {
        setFilter(e.target.value)
      }
      
      // Format date
      def formatDate(date: Date): String = {
        val options = js.Dynamic.literal(
          year = "numeric",
          month = "short",
          day = "numeric"
        )
        date.toLocaleDateString("en-US", options.asInstanceOf[js.Object])
      }
      
      // Get visualization type display name
      def getVizTypeDisplay(vizType: String): String = vizType match {
        case Visualization.KaplanMeier => "Kaplan-Meier Curve"
        case Visualization.HazardRatio => "Hazard Ratio Plot"
        case Visualization.FeatureImportance => "Feature Importance"
        case _ => "Unknown"
      }
      
      // Get visualization type icon
      def getVizTypeIcon(vizType: String): String = vizType match {
        case Visualization.KaplanMeier => "📈"
        case Visualization.HazardRatio => "📊"
        case Visualization.FeatureImportance => "📋"
        case _ => "📄"
      }
      
      // Main content
      val content = React.createElement(
        "div",
        js.Dynamic.literal(
          className = "space-y-6"
        ).asInstanceOf[js.Object],
        
        // Header with actions
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "flex justify-between items-center"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "h1",
            js.Dynamic.literal(
              className = "text-2xl font-bold text-gray-900"
            ).asInstanceOf[js.Object],
            "Visualizations"
          ),
          
          React.createElement(
            "button",
            js.Dynamic.literal(
              className = "px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition duration-150 flex items-center gap-2",
              onClick = () => handleCreateVisualization()
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "span",
              null,
              "+"
            ),
            
            "Create Visualization"
          )
        ),
        
        // Filter and search
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "flex flex-wrap gap-4"
          ).asInstanceOf[js.Object],
          
          // Search box
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "relative flex-grow"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "input",
              js.Dynamic.literal(
                type = "text",
                placeholder = "Search visualizations...",
                className = "w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500",
                value = searchTerm,
                onChange = handleSearchChange
              ).asInstanceOf[js.Object]
            )
          ),
          
          // Filter dropdown
          React.createElement(
            "select",
            js.Dynamic.literal(
              className = "px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500",
              value = filter,
              onChange = handleFilterChange
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "option",
              js.Dynamic.literal(
                value = "all"
              ).asInstanceOf[js.Object],
              "All Types"
            ),
            
            React.createElement(
              "option",
              js.Dynamic.literal(
                value = "kaplan-meier"
              ).asInstanceOf[js.Object],
              "Kaplan-Meier Curves"
            ),
            
            React.createElement(
              "option",
              js.Dynamic.literal(
                value = "hazard-ratio"
              ).asInstanceOf[js.Object],
              "Hazard Ratio Plots"
            ),
            
            React.createElement(
              "option",
              js.Dynamic.literal(
                value = "feature-importance"
              ).asInstanceOf[js.Object],
              "Feature Importance"
            )
          )
        ),
        
        // Error message if present
        error.map(errorMsg => 
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded"
            ).asInstanceOf[js.Object],
            errorMsg
          )
        ).getOrElse(null),
        
        // Loading state
        if (isLoading) {
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "text-center py-10"
            ).asInstanceOf[js.Object],
            "Loading visualizations..."
          )
        } else if (filteredVisualizations.isEmpty) {
          // Empty state
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "text-center py-10 bg-gray-50 rounded-lg"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "p",
              js.Dynamic.literal(
                className = "text-gray-500 mb-4"
              ).asInstanceOf[js.Object],
              "No visualizations found"
            ),
            
            React.createElement(
              "button",
              js.Dynamic.literal(
                className = "px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition duration-150",
                onClick = () => handleCreateVisualization()
              ).asInstanceOf[js.Object],
              "Create your first visualization"
            )
          )
        } else {
          // Visualization grid
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6"
            ).asInstanceOf[js.Object],
            
            filteredVisualizations.map { viz =>
              React.createElement(
                "div",
                js.Dynamic.literal(
                  key = viz.id,
                  className = "bg-white rounded-lg shadow overflow-hidden border border-gray-200 transition duration-150 hover:shadow-md"
                ).asInstanceOf[js.Object],
                
                // Visualization preview
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "h-40 bg-gray-100 flex items-center justify-center p-4 cursor-pointer",
                    onClick = () => handleSelectVisualization(viz.id)
                  ).asInstanceOf[js.Object],
                  
                  React.createElement(
                    "div",
                    js.Dynamic.literal(
                      className = "text-4xl"
                    ).asInstanceOf[js.Object],
                    getVizTypeIcon(viz.vizType)
                  )
                ),
                
                // Card content
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "p-4"
                  ).asInstanceOf[js.Object],
                  
                  // Type badge
                  React.createElement(
                    "span",
                    js.Dynamic.literal(
                      className = "inline-block px-2 py-1 text-xs font-semibold bg-blue-100 text-blue-800 rounded-full mb-2"
                    ).asInstanceOf[js.Object],
                    getVizTypeDisplay(viz.vizType)
                  ),
                  
                  // Title
                  React.createElement(
                    "h3",
                    js.Dynamic.literal(
                      className = "text-lg font-semibold mb-2 cursor-pointer hover:text-blue-600",
                      onClick = () => handleSelectVisualization(viz.id)
                    ).asInstanceOf[js.Object],
                    viz.title
                  ),
                  
                  // Description
                  viz.description.map { desc =>
                    React.createElement(
                      "p",
                      js.Dynamic.literal(
                        className = "text-sm text-gray-600 mb-3 line-clamp-2"
                      ).asInstanceOf[js.Object],
                      desc
                    )
                  }.getOrElse(null),
                  
                  // Info row
                  React.createElement(
                    "div",
                    js.Dynamic.literal(
                      className = "flex justify-between text-xs text-gray-500 mt-4"
                    ).asInstanceOf[js.Object],
                    
                    // Creation date
                    React.createElement(
                      "span",
                      null,
                      s"Created: ${formatDate(viz.createdAt)}"
                    ),
                    
                    // Actions
                    React.createElement(
                      "div",
                      js.Dynamic.literal(
                        className = "flex gap-2"
                      ).asInstanceOf[js.Object],
                      
                      // View button
                      React.createElement(
                        "button",
                        js.Dynamic.literal(
                          className = "text-blue-600 hover:text-blue-800",
                          title = "View",
                          onClick = () => handleSelectVisualization(viz.id)
                        ).asInstanceOf[js.Object],
                        "👁️"
                      ),
                      
                      // Delete button
                      React.createElement(
                        "button",
                        js.Dynamic.literal(
                          className = "text-red-600 hover:text-red-800",
                          title = "Delete",
                          onClick = () => handleDeleteVisualization(viz.id)
                        ).asInstanceOf[js.Object],
                        "🗑️"
                      )
                    )
                  )
                )
              )
            }.toSeq: _*
          )
        }
      )
      
      // Render with layout
      Layout.render(
        Layout.Props(
          children = content,
          title = "Visualizations",
          subtitle = "Create and manage visualizations for survival analysis"
        )
      )
    }
  }
}

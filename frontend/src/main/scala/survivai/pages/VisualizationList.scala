package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Success, Failure}
import survivai.bindings.ReactBindings.*
import survivai.models.{Analysis, Visualization}
import survivai.services.VisualizationService
import survivai.contexts.LayoutContext

object VisualizationList {
  def render(): Element = {
    FC {
      // Set the page title
      val layoutContext = LayoutContext.useLayout()
      layoutContext.setTitle("Visualizations")
      
      // States
      val visualizationsState = React.useState[js.Array[Visualization.Visualization]](js.Array())
      val visualizations = visualizationsState(0).asInstanceOf[js.Array[Visualization.Visualization]]
      val setVisualizations = visualizationsState(1).asInstanceOf[js.Function1[js.Array[Visualization.Visualization], Unit]]
      
      val loadingState = React.useState[Boolean](true)
      val loading = loadingState(0).asInstanceOf[Boolean]
      val setLoading = loadingState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      val errorState = React.useState[Option[String]](None)
      val error = errorState(0).asInstanceOf[Option[String]]
      val setError = errorState(1).asInstanceOf[js.Function1[Option[String], Unit]]
      
      val deleteModalState = React.useState[Option[Visualization.Visualization]](None)
      val deleteModal = deleteModalState(0).asInstanceOf[Option[Visualization.Visualization]]
      val setDeleteModal = deleteModalState(1).asInstanceOf[js.Function1[Option[Visualization.Visualization], Unit]]
      
      val deletingState = React.useState[Boolean](false)
      val deleting = deletingState(0).asInstanceOf[Boolean]
      val setDeleting = deletingState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      // Fetch visualizations on mount
      React.useEffect(() => {
        loadVisualizations()
        () => ()
      }, js.Array())
      
      // Function to load visualizations
      def loadVisualizations(): Unit = {
        setLoading(true)
        setError(None)
        
        VisualizationService.getVisualizations().onComplete {
          case Success(visualizations) =>
            setVisualizations(visualizations.toArray[Visualization.Visualization])
            setLoading(false)
          case Failure(exception) =>
            setError(Some(s"Error loading visualizations: ${exception.getMessage}"))
            setLoading(false)
        }
      }
      
      // Function to handle visualization deletion
      val handleDeleteVisualization = js.Function1 { (viz: Visualization.Visualization) =>
        setDeleteModal(Some(viz))
      }
      
      // Function to confirm deletion
      val confirmDelete = js.Function0 { () =>
        deleteModal.foreach { viz =>
          setDeleting(true)
          
          VisualizationService.deleteVisualization(viz.id).onComplete {
            case Success(_) =>
              // Remove deleted visualization from state
              setVisualizations(visualizations.filter(_.id != viz.id))
              setDeleteModal(None)
              setDeleting(false)
            case Failure(exception) =>
              setError(Some(s"Error deleting visualization: ${exception.getMessage}"))
              setDeleteModal(None)
              setDeleting(false)
          }
        }
      }
      
      // Function to cancel deletion
      val cancelDelete = js.Function0 { () =>
        setDeleteModal(None)
      }
      
      // Format date
      def formatDate(date: js.Date): String = {
        date.toLocaleDateString("en-US", 
          js.Dynamic.literal(
            year = "numeric",
            month = "short",
            day = "numeric"
          ).asInstanceOf[js.Object]
        )
      }
      
      // Function to get visualization type label
      def getTypeLabel(vizType: Visualization.VisualizationType): String = vizType match {
        case Visualization.KaplanMeier => "Kaplan-Meier Curve"
        case Visualization.FeatureImportance => "Feature Importance"
        case Visualization.CumulativeHazard => "Cumulative Hazard"
        case Visualization.StratifiedSurvival => "Stratified Survival"
        case Visualization.TimeDependent => "Time-Dependent Effect"
        case Visualization.Custom => "Custom Visualization"
      }
      
      // Render visualizations list page
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = "space-y-6"
        ).asInstanceOf[js.Object],
        
        // Header with create button
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "flex justify-between items-center"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "h1",
            js.Dynamic.literal(
              className = "text-2xl font-bold"
            ).asInstanceOf[js.Object],
            "Visualizations"
          ),
          
          React.createElement(
            "a",
            js.Dynamic.literal(
              href = "#/visualizations/new",
              className = "bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md inline-flex items-center transition-colors"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "span",
              js.Dynamic.literal(
                className = "mr-2"
              ).asInstanceOf[js.Object],
              "📊"
            ),
            
            "Create New Visualization"
          )
        ),
        
        // Error message
        error.map { errorMsg =>
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-red-50 border border-red-200 text-red-800 rounded-md p-4"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "p",
              js.Dynamic.literal(
                className = "font-medium"
              ).asInstanceOf[js.Object],
              "Error"
            ),
            
            React.createElement(
              "p",
              null,
              errorMsg
            ),
            
            React.createElement(
              "button",
              js.Dynamic.literal(
                onClick = js.Function0(() => loadVisualizations()),
                className = "mt-2 text-red-600 hover:text-red-800 font-medium"
              ).asInstanceOf[js.Object],
              "Try Again"
            )
          )
        }.getOrElse(React.createElement("div", null)),
        
        // Loading indicator
        if (loading) {
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "text-center py-12"
            ).asInstanceOf[js.Object],
            "Loading visualizations..."
          )
        } 
        // Empty state
        else if (visualizations.isEmpty) {
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-white rounded-lg shadow p-8 text-center"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "text-4xl mb-4"
              ).asInstanceOf[js.Object],
              "📊"
            ),
            
            React.createElement(
              "h3",
              js.Dynamic.literal(
                className = "text-lg font-medium mb-2"
              ).asInstanceOf[js.Object],
              "No Visualizations Found"
            ),
            
            React.createElement(
              "p",
              js.Dynamic.literal(
                className = "text-gray-500 mb-6"
              ).asInstanceOf[js.Object],
              "Create your first visualization to explore and communicate your survival analysis results."
            ),
            
            React.createElement(
              "a",
              js.Dynamic.literal(
                href = "#/visualizations/new",
                className = "bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-md inline-flex items-center transition-colors"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "span",
                js.Dynamic.literal(
                  className = "mr-2"
                ).asInstanceOf[js.Object],
                "📊"
              ),
              
              "Create New Visualization"
            )
          )
        } 
        // Grid of visualization cards
        else {
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
            ).asInstanceOf[js.Object],
            
            visualizations.map { viz =>
              React.createElement(
                "div",
                js.Dynamic.literal(
                  key = viz.id,
                  className = "bg-white rounded-lg shadow overflow-hidden hover:shadow-md transition-shadow"
                ).asInstanceOf[js.Object],
                
                // Visualization preview (placeholder)
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "h-48 bg-gray-100 flex items-center justify-center"
                  ).asInstanceOf[js.Object],
                  
                  React.createElement(
                    "div",
                    js.Dynamic.literal(
                      className = "text-4xl text-gray-400"
                    ).asInstanceOf[js.Object],
                    viz.visualizationType match {
                      case Visualization.KaplanMeier => "📈"
                      case Visualization.FeatureImportance => "📊"
                      case Visualization.CumulativeHazard => "📉"
                      case Visualization.StratifiedSurvival => "📊"
                      case Visualization.TimeDependent => "📈"
                      case Visualization.Custom => "📊"
                    }
                  )
                ),
                
                // Visualization details
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "p-4"
                  ).asInstanceOf[js.Object],
                  
                  React.createElement(
                    "h3",
                    js.Dynamic.literal(
                      className = "text-lg font-medium mb-1"
                    ).asInstanceOf[js.Object],
                    viz.title
                  ),
                  
                  React.createElement(
                    "div",
                    js.Dynamic.literal(
                      className = "text-sm text-gray-500 mb-3"
                    ).asInstanceOf[js.Object],
                    getTypeLabel(viz.visualizationType)
                  ),
                  
                  React.createElement(
                    "div",
                    js.Dynamic.literal(
                      className = "text-sm text-gray-500 mb-4"
                    ).asInstanceOf[js.Object],
                    s"Created on ${formatDate(viz.createdAt)}"
                  ),
                  
                  React.createElement(
                    "div",
                    js.Dynamic.literal(
                      className = "flex justify-between"
                    ).asInstanceOf[js.Object],
                    
                    React.createElement(
                      "a",
                      js.Dynamic.literal(
                        href = s"#/visualizations/${viz.id}",
                        className = "text-blue-600 hover:text-blue-800 text-sm font-medium"
                      ).asInstanceOf[js.Object],
                      "View"
                    ),
                    
                    React.createElement(
                      "button",
                      js.Dynamic.literal(
                        onClick = js.Function0(() => handleDeleteVisualization(viz)),
                        className = "text-red-600 hover:text-red-800 text-sm font-medium"
                      ).asInstanceOf[js.Object],
                      "Delete"
                    )
                  )
                )
              )
            }: _*
          )
        },
        
        // Delete confirmation modal
        deleteModal.map { viz =>
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "bg-white rounded-lg shadow-xl p-6 max-w-md w-full"
              ).asInstanceOf[js.Object],
              
              // Modal title
              React.createElement(
                "h3",
                js.Dynamic.literal(
                  className = "text-lg font-medium mb-4"
                ).asInstanceOf[js.Object],
                "Confirm Deletion"
              ),
              
              // Modal body
              React.createElement(
                "p",
                js.Dynamic.literal(
                  className = "mb-6 text-gray-600"
                ).asInstanceOf[js.Object],
                s"Are you sure you want to delete the visualization '${viz.title}'? This action cannot be undone."
              ),
              
              // Modal footer
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "flex justify-end space-x-3"
                ).asInstanceOf[js.Object],
                
                // Cancel button
                React.createElement(
                  "button",
                  js.Dynamic.literal(
                    onClick = cancelDelete,
                    disabled = deleting,
                    className = "px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 transition-colors"
                  ).asInstanceOf[js.Object],
                  "Cancel"
                ),
                
                // Delete button
                React.createElement(
                  "button",
                  js.Dynamic.literal(
                    onClick = confirmDelete,
                    disabled = deleting,
                    className = s"px-4 py-2 bg-red-600 text-white rounded-md ${if (deleting) "opacity-70 cursor-not-allowed" else "hover:bg-red-700"} transition-colors"
                  ).asInstanceOf[js.Object],
                  if (deleting) "Deleting..." else "Delete"
                )
              )
            )
          )
        }.getOrElse(React.createElement("div", null))
      )
    }
  }
}

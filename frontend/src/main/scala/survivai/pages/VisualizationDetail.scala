package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Success, Failure}
import survivai.bindings.ReactBindings.*
import survivai.models.{Analysis, Visualization}
import survivai.services.{AnalysisService, VisualizationService}
import survivai.contexts.LayoutContext
import survivai.components.visualizations.{KaplanMeierCurve, FeatureImportanceViz}

object VisualizationDetail {
  def render(id: String): Element = {
    FC {
      // Set the page title (will be updated with visualization title)
      val layoutContext = LayoutContext.useLayout()
      
      // States
      val visualizationState = React.useState[Option[Visualization.Visualization]](None)
      val visualization = visualizationState(0).asInstanceOf[Option[Visualization.Visualization]]
      val setVisualization = visualizationState(1).asInstanceOf[js.Function1[Option[Visualization.Visualization], Unit]]
      
      val analysisState = React.useState[Option[Analysis.Analysis]](None)
      val analysis = analysisState(0).asInstanceOf[Option[Analysis.Analysis]]
      val setAnalysis = analysisState(1).asInstanceOf[js.Function1[Option[Analysis.Analysis], Unit]]
      
      val loadingState = React.useState[Boolean](true)
      val loading = loadingState(0).asInstanceOf[Boolean]
      val setLoading = loadingState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      val errorState = React.useState[Option[String]](None)
      val error = errorState(0).asInstanceOf[Option[String]]
      val setError = errorState(1).asInstanceOf[js.Function1[Option[String], Unit]]
      
      // Fetch visualization on mount
      React.useEffect(() => {
        fetchVisualization(id)
        () => ()
      }, js.Array(id))
      
      // Function to fetch visualization and related data
      def fetchVisualization(vizId: String): Unit = {
        setLoading(true)
        setError(None)
        
        VisualizationService.getVisualization(vizId).onComplete {
          case Success(viz) =>
            setVisualization(Some(viz))
            layoutContext.setTitle(viz.title)
            
            // Also fetch the related analysis
            AnalysisService.getAnalysis(viz.analysisId).onComplete {
              case Success(analysis) =>
                setAnalysis(Some(analysis))
                setLoading(false)
              case Failure(exception) =>
                setError(Some(s"Error loading analysis data: ${exception.getMessage}"))
                setLoading(false)
            }
            
          case Failure(exception) =>
            setError(Some(s"Error loading visualization: ${exception.getMessage}"))
            setLoading(false)
        }
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
      
      // Render loading state
      if (loading) {
        return React.createElement(
          "div",
          js.Dynamic.literal(
            className = "flex items-center justify-center py-16"
          ).asInstanceOf[js.Object],
          "Loading visualization..."
        )
      }
      
      // Render error state
      error.foreach { err =>
        return React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-red-50 border border-red-200 text-red-800 rounded-md p-4"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "h3",
            js.Dynamic.literal(
              className = "font-medium"
            ).asInstanceOf[js.Object],
            "Error"
          ),
          
          React.createElement(
            "p",
            null,
            err
          ),
          
          React.createElement(
            "button",
            js.Dynamic.literal(
              onClick = js.Function0(() => fetchVisualization(id)),
              className = "mt-2 text-red-600 hover:text-red-800 font-medium"
            ).asInstanceOf[js.Object],
            "Try Again"
          )
        )
      }
      
      // Render not found state
      if (visualization.isEmpty) {
        return React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-yellow-50 border border-yellow-200 text-yellow-800 rounded-md p-4"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "h3",
            js.Dynamic.literal(
              className = "font-medium"
            ).asInstanceOf[js.Object],
            "Visualization Not Found"
          ),
          
          React.createElement(
            "p",
            null,
            s"No visualization found with ID: $id"
          ),
          
          React.createElement(
            "a",
            js.Dynamic.literal(
              href = "#/visualizations",
              className = "mt-2 inline-block text-yellow-600 hover:text-yellow-800"
            ).asInstanceOf[js.Object],
            "Back to Visualizations"
          )
        )
      }
      
      // Get the visualization and analysis objects
      val viz = visualization.get
      val analysisData = analysis.getOrElse(null)
      
      // Render visualization detail
      React.createElement(
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
          
          // Title and subtitle
          React.createElement(
            "div",
            null,
            
            React.createElement(
              "h1",
              js.Dynamic.literal(
                className = "text-2xl font-bold"
              ).asInstanceOf[js.Object],
              viz.title
            ),
            
            React.createElement(
              "p",
              js.Dynamic.literal(
                className = "text-gray-600"
              ).asInstanceOf[js.Object],
              getTypeLabel(viz.visualizationType)
            )
          ),
          
          // Action buttons
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "flex space-x-3"
            ).asInstanceOf[js.Object],
            
            // Back button
            React.createElement(
              "a",
              js.Dynamic.literal(
                href = "#/visualizations",
                className = "px-3 py-1 border border-gray-300 rounded-md text-gray-600 hover:bg-gray-50 transition-colors"
              ).asInstanceOf[js.Object],
              "Back"
            ),
            
            // Edit button
            React.createElement(
              "a",
              js.Dynamic.literal(
                href = s"#/visualizations/edit/${viz.id}",
                className = "px-3 py-1 border border-blue-500 text-blue-500 rounded-md hover:bg-blue-50 transition-colors"
              ).asInstanceOf[js.Object],
              "Edit"
            ),
            
            // Export button
            React.createElement(
              "button",
              js.Dynamic.literal(
                className = "px-3 py-1 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
              ).asInstanceOf[js.Object],
              "Export"
            )
          )
        ),
        
        // Visualization container
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-white rounded-lg shadow p-6"
          ).asInstanceOf[js.Object],
          
          // Description if available
          viz.description.map(desc => 
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "mb-6 text-gray-600"
              ).asInstanceOf[js.Object],
              desc
            )
          ).getOrElse(null),
          
          // Actual visualization based on type
          renderVisualization(viz, analysisData)
        ),
        
        // Metadata card
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-white rounded-lg shadow p-6"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "h3",
            js.Dynamic.literal(
              className = "text-lg font-medium mb-4"
            ).asInstanceOf[js.Object],
            "Visualization Information"
          ),
          
          React.createElement(
            "dl",
            js.Dynamic.literal(
              className = "grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2"
            ).asInstanceOf[js.Object],
            
            // Created date
            infoItem("Created", formatDate(viz.createdAt)),
            
            // Analysis name
            infoItem("Analysis", if (analysisData != null) analysisData.name else viz.analysisId),
            
            // Visualization type
            infoItem("Type", getTypeLabel(viz.visualizationType)),
            
            // Config details
            viz.config match {
              case config: js.Dynamic =>
                // Show relevant configuration options based on visualization type
                Seq(
                  if (js.typeOf(config.colorScheme) != "undefined") {
                    infoItem("Color Scheme", config.colorScheme.asInstanceOf[String])
                  } else null,
                  
                  if (viz.visualizationType == Visualization.StratifiedSurvival && 
                      js.typeOf(config.stratifyBy) != "undefined") {
                    infoItem("Stratify By", config.stratifyBy.asInstanceOf[String])
                  } else null,
                  
                  if (js.typeOf(config.showGrid) != "undefined") {
                    infoItem("Show Grid", config.showGrid.asInstanceOf[Boolean].toString)
                  } else null,
                  
                  if (js.typeOf(config.showLegend) != "undefined") {
                    infoItem("Show Legend", config.showLegend.asInstanceOf[Boolean].toString)
                  } else null
                ).filter(_ != null): _*
              
              case _ => null
            }
          )
        )
      )
    }
  }
  
  // Helper function to render an info item
  private def infoItem(label: String, value: String): Element = {
    React.createElement(
      "div",
      js.Dynamic.literal(
        className = "py-1"
      ).asInstanceOf[js.Object],
      
      React.createElement(
        "dt",
        js.Dynamic.literal(
          className = "text-sm font-medium text-gray-500"
        ).asInstanceOf[js.Object],
        label
      ),
      
      React.createElement(
        "dd",
        js.Dynamic.literal(
          className = "text-sm text-gray-900"
        ).asInstanceOf[js.Object],
        value
      )
    )
  }
  
  // Helper function to render the appropriate visualization based on type
  private def renderVisualization(viz: Visualization.Visualization, analysis: Analysis.Analysis): Element = {
    if (analysis == null) {
      // If analysis data not available, show placeholder
      return React.createElement(
        "div",
        js.Dynamic.literal(
          className = "bg-gray-100 h-72 flex items-center justify-center rounded-md"
        ).asInstanceOf[js.Object],
        
        React.createElement(
          "p",
          js.Dynamic.literal(
            className = "text-gray-500"
          ).asInstanceOf[js.Object],
          "Analysis data not available"
        )
      )
    }
    
    viz.visualizationType match {
      case Visualization.KaplanMeier =>
        // Show Kaplan-Meier curve if survival points are available
        if (analysis.survivalPoints.nonEmpty) {
          KaplanMeierCurve.render(KaplanMeierCurve.Props(
            data = analysis.survivalPoints.toSeq,
            title = viz.title,
            showGrid = viz.config.asInstanceOf[js.Dynamic].showGrid.asInstanceOf[Boolean],
            showLegend = viz.config.asInstanceOf[js.Dynamic].showLegend.asInstanceOf[Boolean],
            colorScheme = viz.config.asInstanceOf[js.Dynamic].colorScheme.asInstanceOf[String]
          ))
        } else {
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-yellow-50 p-4 rounded-md"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "p",
              null,
              "No survival data available for this analysis"
            )
          )
        }
        
      case Visualization.FeatureImportance =>
        // Show feature importance if available
        if (analysis.featureImportance.nonEmpty) {
          FeatureImportanceViz.render(FeatureImportanceViz.Props(
            data = analysis.featureImportance.toSeq,
            showGrid = viz.config.asInstanceOf[js.Dynamic].showGrid.asInstanceOf[Boolean],
            colorScheme = viz.config.asInstanceOf[js.Dynamic].colorScheme.asInstanceOf[String]
          ))
        } else {
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-yellow-50 p-4 rounded-md"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "p",
              null,
              "No feature importance data available for this analysis"
            )
          )
        }
        
      case Visualization.StratifiedSurvival =>
        // This would require custom stratified data, show placeholder for now
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-gray-100 h-72 flex items-center justify-center rounded-md"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "p",
            js.Dynamic.literal(
              className = "text-gray-500"
            ).asInstanceOf[js.Object],
            "Stratified Survival visualization not implemented in this version"
          )
        )
        
      case Visualization.CumulativeHazard =>
        // Show placeholder for cumulative hazard
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-gray-100 h-72 flex items-center justify-center rounded-md"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "p",
            js.Dynamic.literal(
              className = "text-gray-500"
            ).asInstanceOf[js.Object],
            "Cumulative Hazard visualization not implemented in this version"
          )
        )
        
      case Visualization.TimeDependent =>
        // Show placeholder for time-dependent effects
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-gray-100 h-72 flex items-center justify-center rounded-md"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "p",
            js.Dynamic.literal(
              className = "text-gray-500"
            ).asInstanceOf[js.Object],
            "Time-Dependent Effect visualization not implemented in this version"
          )
        )
        
      case _ =>
        // Generic fallback for other visualization types
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-gray-100 h-72 flex items-center justify-center rounded-md"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "p",
            js.Dynamic.literal(
              className = "text-gray-500"
            ).asInstanceOf[js.Object],
            "This visualization type is not supported in the current version"
          )
        )
    }
  }
}

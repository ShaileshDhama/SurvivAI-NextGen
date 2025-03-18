package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Success, Failure}
import survivai.bindings.ReactBindings.*
import survivai.models.{Analysis, Dataset, Visualization}
import survivai.services.{AnalysisService, VisualizationService}
import survivai.contexts.LayoutContext

object CreateVisualization {
  def render(): Element = {
    FC {
      // Set page title
      val layoutContext = LayoutContext.useLayout()
      layoutContext.setTitle("Create Visualization")
      
      // States for form handling
      val titleState = React.useState[String]("")
      val title = titleState(0).asInstanceOf[String]
      val setTitle = titleState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val descriptionState = React.useState[String]("")
      val description = descriptionState(0).asInstanceOf[String]
      val setDescription = descriptionState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val visualizationTypeState = React.useState[Visualization.VisualizationType](Visualization.KaplanMeier)
      val visualizationType = visualizationTypeState(0).asInstanceOf[Visualization.VisualizationType]
      val setVisualizationType = visualizationTypeState(1).asInstanceOf[js.Function1[Visualization.VisualizationType, Unit]]
      
      val analysisIdState = React.useState[String]("")
      val analysisId = analysisIdState(0).asInstanceOf[String]
      val setAnalysisId = analysisIdState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val stratifyByState = React.useState[Option[String]](None)
      val stratifyBy = stratifyByState(0).asInstanceOf[Option[String]]
      val setStratifyBy = stratifyByState(1).asInstanceOf[js.Function1[Option[String], Unit]]
      
      val colorSchemeState = React.useState[String]("blues")
      val colorScheme = colorSchemeState(0).asInstanceOf[String]
      val setColorScheme = colorSchemeState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val showGridState = React.useState[Boolean](true)
      val showGrid = showGridState(0).asInstanceOf[Boolean]
      val setShowGrid = showGridState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      val showLegendState = React.useState[Boolean](true)
      val showLegend = showLegendState(0).asInstanceOf[Boolean]
      val setShowLegend = showLegendState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      // States for API data and interactions
      val analysesState = React.useState[js.Array[Analysis.Analysis]](js.Array())
      val analyses = analysesState(0).asInstanceOf[js.Array[Analysis.Analysis]]
      val setAnalyses = analysesState(1).asInstanceOf[js.Function1[js.Array[Analysis.Analysis], Unit]]
      
      val selectedAnalysisState = React.useState[Option[Analysis.Analysis]](None)
      val selectedAnalysis = selectedAnalysisState(0).asInstanceOf[Option[Analysis.Analysis]]
      val setSelectedAnalysis = selectedAnalysisState(1).asInstanceOf[js.Function1[Option[Analysis.Analysis], Unit]]
      
      val loadingAnalysesState = React.useState[Boolean](true)
      val loadingAnalyses = loadingAnalysesState(0).asInstanceOf[Boolean]
      val setLoadingAnalyses = loadingAnalysesState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      val creatingState = React.useState[Boolean](false)
      val creating = creatingState(0).asInstanceOf[Boolean]
      val setCreating = creatingState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      val errorState = React.useState[Option[String]](None)
      val error = errorState(0).asInstanceOf[Option[String]]
      val setError = errorState(1).asInstanceOf[js.Function1[Option[String], Unit]]
      
      // Fetch analyses on component mount
      React.useEffect(() => {
        fetchAnalyses()
        () => ()
      }, js.Array())
      
      // Fetch completed analyses
      def fetchAnalyses(): Unit = {
        setLoadingAnalyses(true)
        setError(None)
        
        AnalysisService.getAnalyses(Some(js.Array(Analysis.Completed))).onComplete {
          case Success(analyses) =>
            setAnalyses(analyses.toArray[Analysis.Analysis])
            setLoadingAnalyses(false)
          case Failure(exception) =>
            setError(Some(s"Error loading analyses: ${exception.getMessage}"))
            setLoadingAnalyses(false)
        }
      }
      
      // Handle input changes
      val handleTitleChange = js.Function1 { (event: js.Dynamic) =>
        setTitle(event.target.value.asInstanceOf[String])
      }
      
      val handleDescriptionChange = js.Function1 { (event: js.Dynamic) =>
        setDescription(event.target.value.asInstanceOf[String])
      }
      
      val handleVisTypeChange = js.Function1 { (event: js.Dynamic) =>
        val typeValue = event.target.value.asInstanceOf[String]
        setVisualizationType(Visualization.VisualizationType.fromString(typeValue))
      }
      
      val handleAnalysisChange = js.Function1 { (event: js.Dynamic) =>
        val id = event.target.value.asInstanceOf[String]
        setAnalysisId(id)
        
        // Find and set the selected analysis
        val analysis = analyses.find(_.id == id)
        setSelectedAnalysis(analysis)
        
        // Auto-generate title if empty
        if (title.isEmpty && analysis.isDefined) {
          val typeLabel = visualizationType match {
            case Visualization.KaplanMeier => "Kaplan-Meier Curve"
            case Visualization.FeatureImportance => "Feature Importance"
            case Visualization.CumulativeHazard => "Cumulative Hazard"
            case Visualization.StratifiedSurvival => "Stratified Survival"
            case Visualization.TimeDependent => "Time-Dependent Effect"
            case Visualization.Custom => "Visualization"
          }
          setTitle(s"${analysis.get.name} - $typeLabel")
        }
      }
      
      val handleStratifyChange = js.Function1 { (event: js.Dynamic) =>
        val value = event.target.value.asInstanceOf[String]
        setStratifyBy(if (value.isEmpty) None else Some(value))
      }
      
      val handleColorSchemeChange = js.Function1 { (event: js.Dynamic) =>
        setColorScheme(event.target.value.asInstanceOf[String])
      }
      
      val handleShowGridChange = js.Function1 { (event: js.Dynamic) =>
        setShowGrid(event.target.checked.asInstanceOf[Boolean])
      }
      
      val handleShowLegendChange = js.Function1 { (event: js.Dynamic) =>
        setShowLegend(event.target.checked.asInstanceOf[Boolean])
      }
      
      // Handle form submission
      val handleSubmit = js.Function1 { (event: js.Dynamic) =>
        event.preventDefault()
        
        if (title.isEmpty) {
          setError(Some("Title is required"))
          return
        }
        
        if (analysisId.isEmpty) {
          setError(Some("Please select an analysis"))
          return
        }
        
        setCreating(true)
        setError(None)
        
        // Create configuration based on visualization type
        val config = visualizationType match {
          case Visualization.KaplanMeier => 
            js.Dynamic.literal(
              showGrid = showGrid,
              showLegend = showLegend,
              colorScheme = colorScheme
            )
          case Visualization.FeatureImportance =>
            js.Dynamic.literal(
              showGrid = showGrid,
              colorScheme = colorScheme
            )
          case Visualization.StratifiedSurvival =>
            js.Dynamic.literal(
              stratifyBy = stratifyBy.getOrElse(""),
              showGrid = showGrid,
              showLegend = showLegend,
              colorScheme = colorScheme
            )
          case _ =>
            js.Dynamic.literal(
              showGrid = showGrid,
              showLegend = showLegend,
              colorScheme = colorScheme
            )
        }
        
        // Create visualization
        val newVisualization = Visualization.VisualizationCreate(
          title = title,
          description = if (description.isEmpty) None else Some(description),
          analysisId = analysisId,
          visualizationType = visualizationType,
          config = config.asInstanceOf[js.Object]
        )
        
        VisualizationService.createVisualization(newVisualization).onComplete {
          case Success(visualization) =>
            setCreating(false)
            // Redirect to visualization detail page
            js.Dynamic.global.window.location.hash = s"#/visualizations/${visualization.id}"
          case Failure(exception) =>
            setCreating(false)
            setError(Some(s"Error creating visualization: ${exception.getMessage}"))
        }
      }
      
      // Helper function to determine if a field is available for a given visualization type
      def isFieldAvailable(vizType: Visualization.VisualizationType, field: String): Boolean = {
        (vizType, field) match {
          case (Visualization.StratifiedSurvival, "stratifyBy") => true
          case (_, "stratifyBy") => false
          case _ => true
        }
      }
      
      // Render form
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = "space-y-6"
        ).asInstanceOf[js.Object],
        
        // Page header
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
            "Create Visualization"
          ),
          
          React.createElement(
            "a",
            js.Dynamic.literal(
              href = "#/visualizations",
              className = "px-3 py-1 border border-gray-300 rounded-md text-gray-600 hover:bg-gray-50 transition-colors"
            ).asInstanceOf[js.Object],
            "Cancel"
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
              null,
              errorMsg
            )
          )
        }.getOrElse(React.createElement("div", null)),
        
        // Form
        React.createElement(
          "form",
          js.Dynamic.literal(
            onSubmit = handleSubmit,
            className = "bg-white shadow rounded-lg p-6"
          ).asInstanceOf[js.Object],
          
          // Basic information section
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
              "Basic Information"
            ),
            
            // Title field
            React.createElement(
              "div",
              null,
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "title",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "Title *"
              ),
              
              React.createElement(
                "input",
                js.Dynamic.literal(
                  id = "title",
                  name = "title",
                  type = "text",
                  required = true,
                  value = title,
                  onChange = handleTitleChange,
                  className = "appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                ).asInstanceOf[js.Object]
              )
            ),
            
            // Description field
            React.createElement(
              "div",
              null,
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "description",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "Description"
              ),
              
              React.createElement(
                "textarea",
                js.Dynamic.literal(
                  id = "description",
                  name = "description",
                  rows = 3,
                  value = description,
                  onChange = handleDescriptionChange,
                  className = "appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                ).asInstanceOf[js.Object]
              )
            ),
            
            // Visualization type field
            React.createElement(
              "div",
              null,
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "vizType",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "Visualization Type *"
              ),
              
              React.createElement(
                "select",
                js.Dynamic.literal(
                  id = "vizType",
                  name = "vizType",
                  value = Visualization.VisualizationType.toString(visualizationType),
                  onChange = handleVisTypeChange,
                  className = "mt-1 block w-full py-2 px-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = "kaplan-meier"
                  ).asInstanceOf[js.Object],
                  "Kaplan-Meier Curve"
                ),
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = "feature-importance"
                  ).asInstanceOf[js.Object],
                  "Feature Importance"
                ),
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = "cumulative-hazard"
                  ).asInstanceOf[js.Object],
                  "Cumulative Hazard"
                ),
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = "stratified-survival"
                  ).asInstanceOf[js.Object],
                  "Stratified Survival"
                ),
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = "time-dependent"
                  ).asInstanceOf[js.Object],
                  "Time-Dependent Effect"
                )
              )
            ),
            
            // Analysis field
            React.createElement(
              "div",
              null,
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "analysis",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "Analysis *"
              ),
              
              if (loadingAnalyses) {
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "py-2 text-sm text-gray-500"
                  ).asInstanceOf[js.Object],
                  "Loading analyses..."
                )
              } else if (analyses.isEmpty) {
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "py-2 text-sm text-red-500"
                  ).asInstanceOf[js.Object],
                  "No completed analyses available. Complete an analysis first."
                )
              } else {
                React.createElement(
                  "select",
                  js.Dynamic.literal(
                    id = "analysis",
                    name = "analysis",
                    value = analysisId,
                    onChange = handleAnalysisChange,
                    className = "mt-1 block w-full py-2 px-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  ).asInstanceOf[js.Object],
                  
                  React.createElement(
                    "option",
                    js.Dynamic.literal(
                      value = ""
                    ).asInstanceOf[js.Object],
                    "-- Select an analysis --"
                  ),
                  
                  analyses.map { analysis =>
                    React.createElement(
                      "option",
                      js.Dynamic.literal(
                        key = analysis.id,
                        value = analysis.id
                      ).asInstanceOf[js.Object],
                      analysis.name
                    )
                  }: _*
                )
              }
            )
          ),
          
          // Visualization options section
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "space-y-4 mt-8 pt-8 border-t border-gray-200"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "h3",
              js.Dynamic.literal(
                className = "text-lg font-medium"
              ).asInstanceOf[js.Object],
              "Visualization Options"
            ),
            
            // Stratify By field - only for stratified visualizations
            if (isFieldAvailable(visualizationType, "stratifyBy")) {
              React.createElement(
                "div",
                null,
                
                React.createElement(
                  "label",
                  js.Dynamic.literal(
                    htmlFor = "stratifyBy",
                    className = "block text-sm font-medium text-gray-700 mb-1"
                  ).asInstanceOf[js.Object],
                  "Stratify By"
                ),
                
                React.createElement(
                  "select",
                  js.Dynamic.literal(
                    id = "stratifyBy",
                    name = "stratifyBy",
                    value = stratifyBy.getOrElse(""),
                    onChange = handleStratifyChange,
                    className = "mt-1 block w-full py-2 px-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  ).asInstanceOf[js.Object],
                  
                  React.createElement(
                    "option",
                    js.Dynamic.literal(
                      value = ""
                    ).asInstanceOf[js.Object],
                    "-- Select a variable --"
                  ),
                  
                  // Show covariates from selected analysis if available
                  selectedAnalysis.flatMap(_.covariates).getOrElse(js.Array()).map { covariate =>
                    React.createElement(
                      "option",
                      js.Dynamic.literal(
                        key = covariate,
                        value = covariate
                      ).asInstanceOf[js.Object],
                      covariate
                    )
                  }: _*
                )
              )
            } else null,
            
            // Color scheme field
            React.createElement(
              "div",
              null,
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "colorScheme",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "Color Scheme"
              ),
              
              React.createElement(
                "select",
                js.Dynamic.literal(
                  id = "colorScheme",
                  name = "colorScheme",
                  value = colorScheme,
                  onChange = handleColorSchemeChange,
                  className = "mt-1 block w-full py-2 px-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = "blues"
                  ).asInstanceOf[js.Object],
                  "Blues"
                ),
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = "greens"
                  ).asInstanceOf[js.Object],
                  "Greens"
                ),
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = "oranges"
                  ).asInstanceOf[js.Object],
                  "Oranges"
                ),
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = "purples"
                  ).asInstanceOf[js.Object],
                  "Purples"
                ),
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = "spectral"
                  ).asInstanceOf[js.Object],
                  "Spectral"
                ),
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = "viridis"
                  ).asInstanceOf[js.Object],
                  "Viridis"
                )
              )
            ),
            
            // Display options
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "space-y-2"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "flex items-center"
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "input",
                  js.Dynamic.literal(
                    id = "showGrid",
                    name = "showGrid",
                    type = "checkbox",
                    checked = showGrid,
                    onChange = handleShowGridChange,
                    className = "h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  ).asInstanceOf[js.Object]
                ),
                
                React.createElement(
                  "label",
                  js.Dynamic.literal(
                    htmlFor = "showGrid",
                    className = "ml-2 block text-sm text-gray-900"
                  ).asInstanceOf[js.Object],
                  "Show Grid"
                )
              ),
              
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "flex items-center"
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "input",
                  js.Dynamic.literal(
                    id = "showLegend",
                    name = "showLegend",
                    type = "checkbox",
                    checked = showLegend,
                    onChange = handleShowLegendChange,
                    className = "h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  ).asInstanceOf[js.Object]
                ),
                
                React.createElement(
                  "label",
                  js.Dynamic.literal(
                    htmlFor = "showLegend",
                    className = "ml-2 block text-sm text-gray-900"
                  ).asInstanceOf[js.Object],
                  "Show Legend"
                )
              )
            )
          ),
          
          // Submit button
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "pt-8 mt-8 border-t border-gray-200"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "button",
              js.Dynamic.literal(
                type = "submit",
                disabled = creating || loadingAnalyses || analyses.isEmpty,
                className = s"w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${if (creating || loadingAnalyses || analyses.isEmpty) "opacity-70 cursor-not-allowed" else ""}"
              ).asInstanceOf[js.Object],
              if (creating) "Creating..." else "Create Visualization"
            )
          )
        )
      )
    }
  }
}

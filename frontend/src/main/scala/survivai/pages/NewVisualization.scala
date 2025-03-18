package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import survivai.bindings.ReactBindings.*
import survivai.components.layout.Layout
import survivai.services.{AnalysisService, VisualizationService}
import survivai.models.{Analysis, Visualization}
import org.scalajs.dom

object NewVisualization {
  def render(): Element = {
    FC {
      // State hooks
      val (title, setTitle) = useState("")
      val (description, setDescription) = useState("") 
      val (selectedAnalysis, setSelectedAnalysis) = useState[Option[String]](None)
      val (visualizationType, setVisualizationType) = useState(Visualization.KaplanMeier)
      val (analyses, setAnalyses) = useState(js.Array[Analysis.Analysis]())
      val (isLoading, setIsLoading) = useState(false)
      val (isSubmitting, setIsSubmitting) = useState(false)
      val (error, setError) = useState[Option[String]](None)
      val (success, setSuccess) = useState(false)
      
      // Config state for different visualization types
      val (kmConfig, setKmConfig) = useState(js.Dynamic.literal(
        timeVariable = "",
        eventVariable = "",
        groupVariable = ""
      ))
      
      val (hrConfig, setHrConfig) = useState(js.Dynamic.literal(
        covariates = js.Array(),
        sortBySignificance = true
      ))
      
      val (fiConfig, setFiConfig) = useState(js.Dynamic.literal(
        numFeatures = 10,
        sortByImportance = true
      ))
      
      // Fetch analyses on component mount
      useEffect(() => {
        fetchAnalyses()
        () => ()
      }, js.Array())
      
      def fetchAnalyses(): Unit = {
        setIsLoading(true)
        
        AnalysisService.getAnalyses(None).foreach { result =>
          // Filter for completed analyses only
          val completedAnalyses = result.filter(_.status == Analysis.Completed)
          setAnalyses(js.Array(completedAnalyses: _*))
          if (completedAnalyses.nonEmpty) {
            setSelectedAnalysis(Some(completedAnalyses.head.id))
          }
          setIsLoading(false)
        } recover { case ex => 
          console.error("Failed to fetch analyses:", ex.getMessage)
          setError(Some(s"Failed to load analyses: ${ex.getMessage}"))
          setIsLoading(false)
        }
      }
      
      // Handle input changes
      val handleTitleChange = (e: ReactEventFrom[dom.html.Input]) => {
        setTitle(e.target.value)
      }
      
      val handleDescriptionChange = (e: ReactEventFrom[dom.html.TextArea]) => {
        setDescription(e.target.value)
      }
      
      val handleAnalysisChange = (e: ReactEventFrom[dom.html.Select]) => {
        setSelectedAnalysis(Some(e.target.value))
      }
      
      val handleVisualizationTypeChange = (e: ReactEventFrom[dom.html.Select]) => {
        setVisualizationType(e.target.value)
      }
      
      // Handle KM config changes
      val handleKmConfigChange = (field: String, value: String) => {
        val updatedConfig = js.Object.assign(js.Dynamic.literal(), kmConfig)
        js.Object.defineProperty(updatedConfig, field, js.Dynamic.literal(
          value = value,
          writable = true,
          enumerable = true,
          configurable = true
        ))
        setKmConfig(updatedConfig.asInstanceOf[js.Dynamic])
      }
      
      // Handle HR config changes
      val handleHrConfigChange = (field: String, value: js.Any) => {
        val updatedConfig = js.Object.assign(js.Dynamic.literal(), hrConfig)
        js.Object.defineProperty(updatedConfig, field, js.Dynamic.literal(
          value = value,
          writable = true,
          enumerable = true,
          configurable = true
        ))
        setHrConfig(updatedConfig.asInstanceOf[js.Dynamic])
      }
      
      // Handle FI config changes
      val handleFiConfigChange = (field: String, value: js.Any) => {
        val updatedConfig = js.Object.assign(js.Dynamic.literal(), fiConfig)
        js.Object.defineProperty(updatedConfig, field, js.Dynamic.literal(
          value = value,
          writable = true,
          enumerable = true,
          configurable = true
        ))
        setFiConfig(updatedConfig.asInstanceOf[js.Dynamic])
      }
      
      // Determine which config to use based on visualization type
      def getConfigForType(): js.Object = {
        visualizationType match {
          case Visualization.KaplanMeier => kmConfig.asInstanceOf[js.Object]
          case Visualization.HazardRatio => hrConfig.asInstanceOf[js.Object]
          case Visualization.FeatureImportance => fiConfig.asInstanceOf[js.Object]
          case _ => js.Dynamic.literal().asInstanceOf[js.Object]
        }
      }
      
      // Handle form submission
      val handleSubmit = (e: ReactEventFrom[dom.html.Form]) => {
        e.preventDefault()
        
        if (title.trim.isEmpty) {
          setError(Some("Title is required"))
          return
        }
        
        if (selectedAnalysis.isEmpty) {
          setError(Some("Please select an analysis"))
          return
        }
        
        setIsSubmitting(true)
        setError(None)
        
        // Create visualization object
        val newVisualization = js.Dynamic.literal(
          title = title,
          description = if (description.trim.nonEmpty) description else js.undefined,
          analysisId = selectedAnalysis.get,
          visualizationType = visualizationType,
          config = getConfigForType()
        ).asInstanceOf[Visualization.VisualizationCreate]
        
        // Submit to API
        VisualizationService.createVisualization(newVisualization).foreach { result =>
          setSuccess(true)
          setIsSubmitting(false)
          // Redirect to visualization view after short delay
          js.timers.setTimeout(1000) {
            dom.window.location.href = s"#/visualizations/${result.id}"
          }
        } recover { case ex => 
          console.error("Failed to create visualization:", ex.getMessage)
          setError(Some(s"Failed to create visualization: ${ex.getMessage}"))
          setIsSubmitting(false)
        }
      }
      
      // Cancel and go back
      val handleCancel = () => {
        dom.window.location.href = "#/visualizations"
      }
      
      // Different config forms based on visualization type
      def renderConfigForm(): Element = {
        visualizationType match {
          case Visualization.KaplanMeier => 
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "space-y-4"
              ).asInstanceOf[js.Object],
              
              // Time variable field
              React.createElement(
                "div",
                null,
                React.createElement(
                  "label",
                  js.Dynamic.literal(
                    htmlFor = "timeVariable",
                    className = "block text-sm font-medium text-gray-700 mb-1"
                  ).asInstanceOf[js.Object],
                  "Time Variable"
                ),
                React.createElement(
                  "input",
                  js.Dynamic.literal(
                    id = "timeVariable",
                    type = "text",
                    className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                    value = kmConfig.timeVariable,
                    onChange = (e: ReactEventFrom[dom.html.Input]) => handleKmConfigChange("timeVariable", e.target.value)
                  ).asInstanceOf[js.Object]
                )
              ),
              
              // Event variable field
              React.createElement(
                "div",
                null,
                React.createElement(
                  "label",
                  js.Dynamic.literal(
                    htmlFor = "eventVariable",
                    className = "block text-sm font-medium text-gray-700 mb-1"
                  ).asInstanceOf[js.Object],
                  "Event Variable"
                ),
                React.createElement(
                  "input",
                  js.Dynamic.literal(
                    id = "eventVariable",
                    type = "text",
                    className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                    value = kmConfig.eventVariable,
                    onChange = (e: ReactEventFrom[dom.html.Input]) => handleKmConfigChange("eventVariable", e.target.value)
                  ).asInstanceOf[js.Object]
                )
              ),
              
              // Group variable field
              React.createElement(
                "div",
                null,
                React.createElement(
                  "label",
                  js.Dynamic.literal(
                    htmlFor = "groupVariable",
                    className = "block text-sm font-medium text-gray-700 mb-1"
                  ).asInstanceOf[js.Object],
                  "Group Variable (optional)"
                ),
                React.createElement(
                  "input",
                  js.Dynamic.literal(
                    id = "groupVariable",
                    type = "text",
                    className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                    value = kmConfig.groupVariable,
                    onChange = (e: ReactEventFrom[dom.html.Input]) => handleKmConfigChange("groupVariable", e.target.value)
                  ).asInstanceOf[js.Object]
                )
              )
            )
            
          case Visualization.HazardRatio =>
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "space-y-4"
              ).asInstanceOf[js.Object],
              
              // Sort by significance checkbox
              React.createElement(
                "div",
                null,
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "flex items-center"
                  ).asInstanceOf[js.Object],
                  React.createElement(
                    "input",
                    js.Dynamic.literal(
                      id = "sortBySignificance",
                      type = "checkbox",
                      className = "h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded",
                      checked = hrConfig.sortBySignificance,
                      onChange = (e: ReactEventFrom[dom.html.Input]) => handleHrConfigChange("sortBySignificance", e.target.checked)
                    ).asInstanceOf[js.Object]
                  ),
                  React.createElement(
                    "label",
                    js.Dynamic.literal(
                      htmlFor = "sortBySignificance",
                      className = "ml-2 block text-sm text-gray-900"
                    ).asInstanceOf[js.Object],
                    "Sort by significance"
                  )
                )
              )
            )
            
          case Visualization.FeatureImportance =>
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "space-y-4"
              ).asInstanceOf[js.Object],
              
              // Number of features field
              React.createElement(
                "div",
                null,
                React.createElement(
                  "label",
                  js.Dynamic.literal(
                    htmlFor = "numFeatures",
                    className = "block text-sm font-medium text-gray-700 mb-1"
                  ).asInstanceOf[js.Object],
                  "Number of Features"
                ),
                React.createElement(
                  "input",
                  js.Dynamic.literal(
                    id = "numFeatures",
                    type = "number",
                    min = 1,
                    max = 50,
                    className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                    value = fiConfig.numFeatures,
                    onChange = (e: ReactEventFrom[dom.html.Input]) => handleFiConfigChange("numFeatures", e.target.valueAsNumber)
                  ).asInstanceOf[js.Object]
                )
              ),
              
              // Sort by importance checkbox
              React.createElement(
                "div",
                null,
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "flex items-center"
                  ).asInstanceOf[js.Object],
                  React.createElement(
                    "input",
                    js.Dynamic.literal(
                      id = "sortByImportance",
                      type = "checkbox",
                      className = "h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded",
                      checked = fiConfig.sortByImportance,
                      onChange = (e: ReactEventFrom[dom.html.Input]) => handleFiConfigChange("sortByImportance", e.target.checked)
                    ).asInstanceOf[js.Object]
                  ),
                  React.createElement(
                    "label",
                    js.Dynamic.literal(
                      htmlFor = "sortByImportance",
                      className = "ml-2 block text-sm text-gray-900"
                    ).asInstanceOf[js.Object],
                    "Sort by importance"
                  )
                )
              )
            )
            
          case _ => null
        }
      }
      
      // Main content
      val content = React.createElement(
        "div",
        js.Dynamic.literal(
          className = "max-w-3xl mx-auto"
        ).asInstanceOf[js.Object],
        
        // Form header
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "md:flex md:items-center md:justify-between mb-6"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "h2",
            js.Dynamic.literal(
              className = "text-2xl font-bold text-gray-900"
            ).asInstanceOf[js.Object],
            "Create New Visualization"
          )
        ),
        
        // Error message if present
        error.map(errorMsg => 
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4"
            ).asInstanceOf[js.Object],
            errorMsg
          )
        ).getOrElse(null),
        
        // Success message if present
        if (success) {
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded mb-4"
            ).asInstanceOf[js.Object],
            "Visualization created successfully! Redirecting..."
          )
        } else null,
        
        // Form
        React.createElement(
          "form",
          js.Dynamic.literal(
            onSubmit = handleSubmit,
            className = "bg-white shadow rounded-lg p-6"
          ).asInstanceOf[js.Object],
          
          // Main form fields
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "space-y-6"
            ).asInstanceOf[js.Object],
            
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
                  type = "text",
                  className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                  value = title,
                  onChange = handleTitleChange,
                  required = true
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
                  className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                  rows = 3,
                  value = description,
                  onChange = handleDescriptionChange
                ).asInstanceOf[js.Object]
              )
            ),
            
            // Analysis selection
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
              if (isLoading) {
                React.createElement(
                  "p",
                  js.Dynamic.literal(
                    className = "text-sm text-gray-500"
                  ).asInstanceOf[js.Object],
                  "Loading analyses..."
                )
              } else if (analyses.isEmpty) {
                React.createElement(
                  "p",
                  js.Dynamic.literal(
                    className = "text-sm text-red-500"
                  ).asInstanceOf[js.Object],
                  "No completed analyses available. Please complete an analysis first."
                )
              } else {
                React.createElement(
                  "select",
                  js.Dynamic.literal(
                    id = "analysis",
                    className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                    value = selectedAnalysis.getOrElse(""),
                    onChange = handleAnalysisChange,
                    required = true
                  ).asInstanceOf[js.Object],
                  
                  analyses.map { analysis =>
                    React.createElement(
                      "option",
                      js.Dynamic.literal(
                        key = analysis.id,
                        value = analysis.id
                      ).asInstanceOf[js.Object],
                      analysis.name
                    )
                  }.toSeq: _*
                )
              }
            ),
            
            // Visualization type selection
            React.createElement(
              "div",
              null,
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "visualizationType",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "Visualization Type *"
              ),
              React.createElement(
                "select",
                js.Dynamic.literal(
                  id = "visualizationType",
                  className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                  value = visualizationType,
                  onChange = handleVisualizationTypeChange,
                  required = true
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = Visualization.KaplanMeier
                  ).asInstanceOf[js.Object],
                  "Kaplan-Meier Curve"
                ),
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = Visualization.HazardRatio
                  ).asInstanceOf[js.Object],
                  "Hazard Ratio Plot"
                ),
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = Visualization.FeatureImportance
                  ).asInstanceOf[js.Object],
                  "Feature Importance"
                )
              )
            ),
            
            // Configuration fields based on visualization type
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "border-t border-gray-200 pt-4 mt-4"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "h3",
                js.Dynamic.literal(
                  className = "text-lg font-medium text-gray-900 mb-4"
                ).asInstanceOf[js.Object],
                "Visualization Configuration"
              ),
              
              renderConfigForm()
            ),
            
            // Form actions
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "flex items-center justify-end space-x-3 pt-4 border-t border-gray-200 mt-6"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "button",
                js.Dynamic.literal(
                  type = "button",
                  className = "px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2",
                  onClick = handleCancel
                ).asInstanceOf[js.Object],
                "Cancel"
              ),
              
              React.createElement(
                "button",
                js.Dynamic.literal(
                  type = "submit",
                  className = "px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2",
                  disabled = isSubmitting
                ).asInstanceOf[js.Object],
                if (isSubmitting) "Creating..." else "Create Visualization"
              )
            )
          )
        )
      )
      
      // Render with layout
      Layout.render(
        Layout.Props(
          children = content,
          title = "Create Visualization",
          subtitle = "Create a new visualization for your survival analysis"
        )
      )
    }
  }
}

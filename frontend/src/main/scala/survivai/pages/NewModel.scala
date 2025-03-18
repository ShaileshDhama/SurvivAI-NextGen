package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import survivai.bindings.ReactBindings.*
import survivai.components.layout.Layout
import survivai.services.{AnalysisService, ModelService}
import survivai.models.{Analysis, Model}
import org.scalajs.dom

object NewModel {
  def render(): Element = {
    FC {
      // State hooks
      val (name, setName) = useState("")
      val (description, setDescription) = useState("") 
      val (selectedAnalysis, setSelectedAnalysis) = useState[Option[String]](None)
      val (modelType, setModelType) = useState(Model.ModelType.Cox)
      val (analyses, setAnalyses) = useState(js.Array[Analysis.Analysis]())
      val (isLoading, setIsLoading) = useState(false)
      val (isSubmitting, setIsSubmitting) = useState(false)
      val (error, setError) = useState[Option[String]](None)
      val (success, setSuccess) = useState(false)
      
      // Parameters state for different model types
      val (coxParameters, setCoxParameters) = useState(js.Dynamic.literal(
        penalizer = 0.1,
        alpha = 0.05,
        tie_method = "efron"
      ))
      
      val (rsfParameters, setRsfParameters) = useState(js.Dynamic.literal(
        n_estimators = 100,
        max_depth = 5,
        min_samples_split = 2,
        max_features = "sqrt"
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
      val handleNameChange = (e: ReactEventFrom[dom.html.Input]) => {
        setName(e.target.value)
      }
      
      val handleDescriptionChange = (e: ReactEventFrom[dom.html.TextArea]) => {
        setDescription(e.target.value)
      }
      
      val handleAnalysisChange = (e: ReactEventFrom[dom.html.Select]) => {
        setSelectedAnalysis(Some(e.target.value))
      }
      
      val handleModelTypeChange = (e: ReactEventFrom[dom.html.Select]) => {
        val newType = e.target.value match {
          case Model.CoxPH => Model.ModelType.Cox
          case Model.RandomSurvivalForest => Model.ModelType.RSF
          case Model.KaplanMeier => Model.ModelType.KM
          case _ => Model.ModelType.Cox
        }
        setModelType(newType)
      }
      
      // Handle Cox parameters changes
      val handleCoxParameterChange = (field: String, value: js.Any) => {
        val updatedParams = js.Object.assign(js.Dynamic.literal(), coxParameters)
        js.Object.defineProperty(updatedParams, field, js.Dynamic.literal(
          value = value,
          writable = true,
          enumerable = true,
          configurable = true
        ))
        setCoxParameters(updatedParams.asInstanceOf[js.Dynamic])
      }
      
      // Handle RSF parameters changes
      val handleRsfParameterChange = (field: String, value: js.Any) => {
        val updatedParams = js.Object.assign(js.Dynamic.literal(), rsfParameters)
        js.Object.defineProperty(updatedParams, field, js.Dynamic.literal(
          value = value,
          writable = true,
          enumerable = true,
          configurable = true
        ))
        setRsfParameters(updatedParams.asInstanceOf[js.Dynamic])
      }
      
      // Determine which parameters to use based on model type
      def getParametersForType(): js.Object = {
        modelType match {
          case Model.ModelType.Cox => coxParameters.asInstanceOf[js.Object]
          case Model.ModelType.RSF => rsfParameters.asInstanceOf[js.Object]
          case Model.ModelType.KM => js.Dynamic.literal().asInstanceOf[js.Object] // KM doesn't need parameters
          case _ => js.Dynamic.literal().asInstanceOf[js.Object]
        }
      }
      
      // Handle form submission
      val handleSubmit = (e: ReactEventFrom[dom.html.Form]) => {
        e.preventDefault()
        
        if (name.trim.isEmpty) {
          setError(Some("Name is required"))
          return
        }
        
        if (selectedAnalysis.isEmpty) {
          setError(Some("Please select an analysis"))
          return
        }
        
        setIsSubmitting(true)
        setError(None)
        
        // Create model object
        val newModel = js.Dynamic.literal(
          name = name,
          description = if (description.trim.nonEmpty) description else js.undefined,
          analysisId = selectedAnalysis.get,
          modelType = modelType,
          parameters = getParametersForType()
        ).asInstanceOf[Model.ModelCreate]
        
        // Submit to API
        ModelService.createModel(newModel).foreach { result =>
          setSuccess(true)
          setIsSubmitting(false)
          // Redirect to model view after short delay
          js.timers.setTimeout(1000) {
            dom.window.location.href = s"#/models/${result.id}"
          }
        } recover { case ex => 
          console.error("Failed to create model:", ex.getMessage)
          setError(Some(s"Failed to create model: ${ex.getMessage}"))
          setIsSubmitting(false)
        }
      }
      
      // Cancel and go back
      val handleCancel = () => {
        dom.window.location.href = "#/models"
      }
      
      // Different parameter forms based on model type
      def renderParametersForm(): Element = {
        modelType match {
          case Model.ModelType.Cox => 
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "space-y-4"
              ).asInstanceOf[js.Object],
              
              // Penalizer field
              React.createElement(
                "div",
                null,
                React.createElement(
                  "label",
                  js.Dynamic.literal(
                    htmlFor = "penalizer",
                    className = "block text-sm font-medium text-gray-700 mb-1"
                  ).asInstanceOf[js.Object],
                  "Penalizer"
                ),
                React.createElement(
                  "input",
                  js.Dynamic.literal(
                    id = "penalizer",
                    type = "number",
                    step = "0.01",
                    min = "0",
                    max = "1",
                    className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                    value = coxParameters.penalizer,
                    onChange = (e: ReactEventFrom[dom.html.Input]) => handleCoxParameterChange("penalizer", e.target.valueAsNumber)
                  ).asInstanceOf[js.Object]
                )
              ),
              
              // Alpha field
              React.createElement(
                "div",
                null,
                React.createElement(
                  "label",
                  js.Dynamic.literal(
                    htmlFor = "alpha",
                    className = "block text-sm font-medium text-gray-700 mb-1"
                  ).asInstanceOf[js.Object],
                  "Alpha (Significance level)"
                ),
                React.createElement(
                  "input",
                  js.Dynamic.literal(
                    id = "alpha",
                    type = "number",
                    step = "0.01",
                    min = "0.01",
                    max = "0.5",
                    className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                    value = coxParameters.alpha,
                    onChange = (e: ReactEventFrom[dom.html.Input]) => handleCoxParameterChange("alpha", e.target.valueAsNumber)
                  ).asInstanceOf[js.Object]
                )
              ),
              
              // Tie method field
              React.createElement(
                "div",
                null,
                React.createElement(
                  "label",
                  js.Dynamic.literal(
                    htmlFor = "tie_method",
                    className = "block text-sm font-medium text-gray-700 mb-1"
                  ).asInstanceOf[js.Object],
                  "Tie Method"
                ),
                React.createElement(
                  "select",
                  js.Dynamic.literal(
                    id = "tie_method",
                    className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                    value = coxParameters.tie_method,
                    onChange = (e: ReactEventFrom[dom.html.Select]) => handleCoxParameterChange("tie_method", e.target.value)
                  ).asInstanceOf[js.Object],
                  
                  React.createElement(
                    "option",
                    js.Dynamic.literal(
                      value = "efron"
                    ).asInstanceOf[js.Object],
                    "Efron"
                  ),
                  
                  React.createElement(
                    "option",
                    js.Dynamic.literal(
                      value = "breslow"
                    ).asInstanceOf[js.Object],
                    "Breslow"
                  )
                )
              )
            )
            
          case Model.ModelType.RSF =>
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "space-y-4"
              ).asInstanceOf[js.Object],
              
              // n_estimators field
              React.createElement(
                "div",
                null,
                React.createElement(
                  "label",
                  js.Dynamic.literal(
                    htmlFor = "n_estimators",
                    className = "block text-sm font-medium text-gray-700 mb-1"
                  ).asInstanceOf[js.Object],
                  "Number of Estimators"
                ),
                React.createElement(
                  "input",
                  js.Dynamic.literal(
                    id = "n_estimators",
                    type = "number",
                    min = "10",
                    max = "1000",
                    step = "10",
                    className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                    value = rsfParameters.n_estimators,
                    onChange = (e: ReactEventFrom[dom.html.Input]) => handleRsfParameterChange("n_estimators", e.target.valueAsNumber)
                  ).asInstanceOf[js.Object]
                )
              ),
              
              // max_depth field
              React.createElement(
                "div",
                null,
                React.createElement(
                  "label",
                  js.Dynamic.literal(
                    htmlFor = "max_depth",
                    className = "block text-sm font-medium text-gray-700 mb-1"
                  ).asInstanceOf[js.Object],
                  "Maximum Depth"
                ),
                React.createElement(
                  "input",
                  js.Dynamic.literal(
                    id = "max_depth",
                    type = "number",
                    min = "1",
                    max = "30",
                    className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                    value = rsfParameters.max_depth,
                    onChange = (e: ReactEventFrom[dom.html.Input]) => handleRsfParameterChange("max_depth", e.target.valueAsNumber)
                  ).asInstanceOf[js.Object]
                )
              ),
              
              // min_samples_split field
              React.createElement(
                "div",
                null,
                React.createElement(
                  "label",
                  js.Dynamic.literal(
                    htmlFor = "min_samples_split",
                    className = "block text-sm font-medium text-gray-700 mb-1"
                  ).asInstanceOf[js.Object],
                  "Minimum Samples to Split"
                ),
                React.createElement(
                  "input",
                  js.Dynamic.literal(
                    id = "min_samples_split",
                    type = "number",
                    min = "2",
                    max = "20",
                    className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                    value = rsfParameters.min_samples_split,
                    onChange = (e: ReactEventFrom[dom.html.Input]) => handleRsfParameterChange("min_samples_split", e.target.valueAsNumber)
                  ).asInstanceOf[js.Object]
                )
              ),
              
              // max_features field
              React.createElement(
                "div",
                null,
                React.createElement(
                  "label",
                  js.Dynamic.literal(
                    htmlFor = "max_features",
                    className = "block text-sm font-medium text-gray-700 mb-1"
                  ).asInstanceOf[js.Object],
                  "Maximum Features"
                ),
                React.createElement(
                  "select",
                  js.Dynamic.literal(
                    id = "max_features",
                    className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                    value = rsfParameters.max_features,
                    onChange = (e: ReactEventFrom[dom.html.Select]) => handleRsfParameterChange("max_features", e.target.value)
                  ).asInstanceOf[js.Object],
                  
                  React.createElement(
                    "option",
                    js.Dynamic.literal(
                      value = "sqrt"
                    ).asInstanceOf[js.Object],
                    "Square Root"
                  ),
                  
                  React.createElement(
                    "option",
                    js.Dynamic.literal(
                      value = "log2"
                    ).asInstanceOf[js.Object],
                    "Log2"
                  ),
                  
                  React.createElement(
                    "option",
                    js.Dynamic.literal(
                      value = "auto"
                    ).asInstanceOf[js.Object],
                    "Auto"
                  )
                )
              )
            )
            
          case Model.ModelType.KM =>
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "p-4 bg-gray-50 rounded-md"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "p",
                js.Dynamic.literal(
                  className = "text-sm text-gray-600"
                ).asInstanceOf[js.Object],
                "Kaplan-Meier models don't require additional parameters. The model will be built using the time and event variables from the selected analysis."
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
            "Create New Model"
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
            "Model created successfully! Redirecting..."
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
                "Name *"
              ),
              React.createElement(
                "input",
                js.Dynamic.literal(
                  id = "name",
                  type = "text",
                  className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                  value = name,
                  onChange = handleNameChange,
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
            
            // Model type selection
            React.createElement(
              "div",
              null,
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "modelType",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "Model Type *"
              ),
              React.createElement(
                "select",
                js.Dynamic.literal(
                  id = "modelType",
                  className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                  value = Model.ModelType.toString(modelType),
                  onChange = handleModelTypeChange,
                  required = true
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = Model.CoxPH
                  ).asInstanceOf[js.Object],
                  "Cox Proportional Hazards"
                ),
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = Model.RandomSurvivalForest
                  ).asInstanceOf[js.Object],
                  "Random Survival Forest"
                ),
                
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = Model.KaplanMeier
                  ).asInstanceOf[js.Object],
                  "Kaplan-Meier"
                )
              )
            ),
            
            // Parameters fields based on model type
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
                "Model Parameters"
              ),
              
              renderParametersForm()
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
                if (isSubmitting) "Creating..." else "Create Model"
              )
            )
          )
        )
      )
      
      // Render with layout
      Layout.render(
        Layout.Props(
          children = content,
          title = "Create Model",
          subtitle = "Create a new survival analysis model"
        )
      )
    }
  }
}

package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import survivai.bindings.ReactBindings.*
import survivai.components.layout.Layout
import survivai.services.ModelService
import survivai.models.Model
import java.util.Date
import org.scalajs.dom

object Models {
  def render(): Element = {
    FC {
      // State hooks
      val (models, setModels) = useState(js.Array[Model.Model]())
      val (isLoading, setIsLoading) = useState(true)
      val (searchTerm, setSearchTerm) = useState("")
      val (typeFilter, setTypeFilter) = useState("all")
      val (error, setError) = useState[Option[String]](None)
      
      // Fetch models on component mount
      useEffect(() => {
        fetchModels()
        () => ()
      }, js.Array())
      
      // Fetch models function
      def fetchModels(): Unit = {
        setIsLoading(true)
        setError(None)
        
        ModelService.getModels(None).foreach { result =>
          setModels(js.Array(result: _*))
          setIsLoading(false)
        } recover { case ex => 
          console.error("Failed to fetch models:", ex.getMessage)
          setError(Some(s"Failed to load models: ${ex.getMessage}"))
          setIsLoading(false)
        }
      }
      
      // Filter models based on search term and type filter
      val filteredModels = models.filter { model =>
        val matchesSearch = searchTerm.isEmpty || 
          model.name.toLowerCase.contains(searchTerm.toLowerCase) || 
          model.description.exists(_.toLowerCase.contains(searchTerm.toLowerCase))
        
        val matchesFilter = typeFilter match {
          case "all" => true
          case "cox" => model.modelType == Model.CoxPH
          case "rsf" => model.modelType == Model.RandomSurvivalForest
          case "km" => model.modelType == Model.KaplanMeier
          case _ => true
        }
        
        matchesSearch && matchesFilter
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
      
      // Get model type display name
      def getModelTypeDisplay(modelType: String): String = modelType match {
        case Model.CoxPH => "Cox Proportional Hazards"
        case Model.RandomSurvivalForest => "Random Survival Forest"
        case Model.KaplanMeier => "Kaplan-Meier"
        case _ => "Unknown"
      }
      
      // Handle model selection
      val handleSelectModel = (modelId: String) => {
        dom.window.location.href = s"#/models/${modelId}"
      }
      
      // Handle create new model
      val handleCreateModel = () => {
        dom.window.location.href = "#/models/new"
      }
      
      // Handle delete model
      val handleDeleteModel = (modelId: String) => {
        if (dom.window.confirm("Are you sure you want to delete this model? This action cannot be undone.")) {
          ModelService.deleteModel(modelId).foreach { success =>
            if (success) {
              fetchModels()
            } else {
              setError(Some("Failed to delete model"))
            }
          }
        }
      }
      
      // Handle search term change
      val handleSearchChange = (e: ReactEventFrom[dom.html.Input]) => {
        setSearchTerm(e.target.value)
      }
      
      // Handle type filter change
      val handleTypeFilterChange = (e: ReactEventFrom[dom.html.Select]) => {
        setTypeFilter(e.target.value)
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
            "Models"
          ),
          
          React.createElement(
            "button",
            js.Dynamic.literal(
              className = "px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition duration-150 flex items-center gap-2",
              onClick = () => handleCreateModel()
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "span",
              null,
              "+"
            ),
            
            "Create Model"
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
                placeholder = "Search models...",
                className = "w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500",
                value = searchTerm,
                onChange = handleSearchChange
              ).asInstanceOf[js.Object]
            )
          ),
          
          // Type filter dropdown
          React.createElement(
            "select",
            js.Dynamic.literal(
              className = "px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500",
              value = typeFilter,
              onChange = handleTypeFilterChange
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
                value = "cox"
              ).asInstanceOf[js.Object],
              "Cox Proportional Hazards"
            ),
            
            React.createElement(
              "option",
              js.Dynamic.literal(
                value = "rsf"
              ).asInstanceOf[js.Object],
              "Random Survival Forest"
            ),
            
            React.createElement(
              "option",
              js.Dynamic.literal(
                value = "km"
              ).asInstanceOf[js.Object],
              "Kaplan-Meier"
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
            "Loading models..."
          )
        } else if (filteredModels.isEmpty) {
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
              "No models found"
            ),
            
            React.createElement(
              "button",
              js.Dynamic.literal(
                className = "px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition duration-150",
                onClick = () => handleCreateModel()
              ).asInstanceOf[js.Object],
              "Create your first model"
            )
          )
        } else {
          // Models card grid
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6"
            ).asInstanceOf[js.Object],
            
            filteredModels.map { model =>
              React.createElement(
                "div",
                js.Dynamic.literal(
                  key = model.id,
                  className = "bg-white rounded-lg shadow overflow-hidden border border-gray-200 transition duration-150 hover:shadow-md"
                ).asInstanceOf[js.Object],
                
                // Model header
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "p-4 border-b border-gray-200"
                  ).asInstanceOf[js.Object],
                  
                  // Model type badge
                  React.createElement(
                    "span",
                    js.Dynamic.literal(
                      className = "inline-block px-2 py-1 text-xs font-semibold bg-blue-100 text-blue-800 rounded-full mb-2"
                    ).asInstanceOf[js.Object],
                    getModelTypeDisplay(model.modelType)
                  ),
                  
                  // Title
                  React.createElement(
                    "h3",
                    js.Dynamic.literal(
                      className = "text-lg font-semibold mb-2 cursor-pointer hover:text-blue-600",
                      onClick = () => handleSelectModel(model.id)
                    ).asInstanceOf[js.Object],
                    model.name
                  ),
                  
                  // Description
                  model.description.map { desc =>
                    React.createElement(
                      "p",
                      js.Dynamic.literal(
                        className = "text-sm text-gray-600 mb-3 line-clamp-2"
                      ).asInstanceOf[js.Object],
                      desc
                    )
                  }.getOrElse(null)
                ),
                
                // Model metrics
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "p-4 bg-gray-50"
                  ).asInstanceOf[js.Object],
                  
                  // Metrics grid
                  React.createElement(
                    "div",
                    js.Dynamic.literal(
                      className = "grid grid-cols-2 gap-4"
                    ).asInstanceOf[js.Object],
                    
                    // Accuracy metric
                    React.createElement(
                      "div",
                      js.Dynamic.literal(
                        className = "text-center"
                      ).asInstanceOf[js.Object],
                      
                      React.createElement(
                        "div",
                        js.Dynamic.literal(
                          className = "text-xs text-gray-500 mb-1"
                        ).asInstanceOf[js.Object],
                        "C-Index"
                      ),
                      
                      React.createElement(
                        "div",
                        js.Dynamic.literal(
                          className = "text-lg font-semibold"
                        ).asInstanceOf[js.Object],
                        model.metrics.cIndex.map(c => f"${c}%.3f").getOrElse("N/A")
                      )
                    ),
                    
                    // Training time metric
                    React.createElement(
                      "div",
                      js.Dynamic.literal(
                        className = "text-center"
                      ).asInstanceOf[js.Object],
                      
                      React.createElement(
                        "div",
                        js.Dynamic.literal(
                          className = "text-xs text-gray-500 mb-1"
                        ).asInstanceOf[js.Object],
                        "Training Time"
                      ),
                      
                      React.createElement(
                        "div",
                        js.Dynamic.literal(
                          className = "text-lg font-semibold"
                        ).asInstanceOf[js.Object],
                        s"${model.trainingTime}s"
                      )
                    )
                  )
                ),
                
                // Card footer
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "p-4 border-t border-gray-200 flex justify-between items-center"
                  ).asInstanceOf[js.Object],
                  
                  // Creation date
                  React.createElement(
                    "span",
                    js.Dynamic.literal(
                      className = "text-xs text-gray-500"
                    ).asInstanceOf[js.Object],
                    s"Created: ${formatDate(model.createdAt)}"
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
                        onClick = () => handleSelectModel(model.id)
                      ).asInstanceOf[js.Object],
                      "ud83dudc41ufe0f"
                    ),
                    
                    // Delete button
                    React.createElement(
                      "button",
                      js.Dynamic.literal(
                        className = "text-red-600 hover:text-red-800",
                        title = "Delete",
                        onClick = (e: js.Dynamic) => {
                          e.stopPropagation()
                          handleDeleteModel(model.id)
                        }
                      ).asInstanceOf[js.Object],
                      "ud83duddd1ufe0f"
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
          title = "Models",
          subtitle = "Manage your survival analysis models"
        )
      )
    }
  }
}

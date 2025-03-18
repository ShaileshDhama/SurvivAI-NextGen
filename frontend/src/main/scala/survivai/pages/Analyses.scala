package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import survivai.bindings.ReactBindings.*
import survivai.components.layout.Layout
import survivai.services.AnalysisService
import survivai.models.Analysis
import java.util.Date
import org.scalajs.dom

object Analyses {
  def render(): Element = {
    FC {
      // State hooks
      val (analyses, setAnalyses) = useState(js.Array[Analysis.Analysis]())
      val (isLoading, setIsLoading) = useState(true)
      val (searchTerm, setSearchTerm) = useState("")
      val (statusFilter, setStatusFilter) = useState("all")
      val (error, setError) = useState[Option[String]](None)
      
      // Fetch analyses on component mount
      useEffect(() => {
        fetchAnalyses()
        () => ()
      }, js.Array())
      
      // Fetch analyses function
      def fetchAnalyses(): Unit = {
        setIsLoading(true)
        setError(None)
        
        AnalysisService.getAnalyses(None).foreach { result =>
          setAnalyses(js.Array(result: _*))
          setIsLoading(false)
        } recover { case ex => 
          console.error("Failed to fetch analyses:", ex.getMessage)
          setError(Some(s"Failed to load analyses: ${ex.getMessage}"))
          setIsLoading(false)
        }
      }
      
      // Filter analyses based on search term and status filter
      val filteredAnalyses = analyses.filter { analysis =>
        val matchesSearch = searchTerm.isEmpty || 
          analysis.name.toLowerCase.contains(searchTerm.toLowerCase) || 
          analysis.description.exists(_.toLowerCase.contains(searchTerm.toLowerCase))
        
        val matchesFilter = statusFilter match {
          case "all" => true
          case "completed" => analysis.status == Analysis.Completed
          case "running" => analysis.status == Analysis.Running
          case "pending" => analysis.status == Analysis.Pending
          case "failed" => analysis.status == Analysis.Failed
          case _ => true
        }
        
        matchesSearch && matchesFilter
      }
      
      // Handle analysis selection
      val handleSelectAnalysis = (analysisId: String) => {
        dom.window.location.href = s"#/analyses/${analysisId}"
      }
      
      // Handle create new analysis
      val handleCreateAnalysis = () => {
        dom.window.location.href = "#/analyses/new"
      }
      
      // Handle delete analysis
      val handleDeleteAnalysis = (analysisId: String) => {
        if (dom.window.confirm("Are you sure you want to delete this analysis? This action cannot be undone.")) {
          AnalysisService.deleteAnalysis(analysisId).foreach { success =>
            if (success) {
              fetchAnalyses()
            } else {
              setError(Some("Failed to delete analysis"))
            }
          }
        }
      }
      
      // Handle run analysis
      val handleRunAnalysis = (analysisId: String) => {
        AnalysisService.runAnalysis(analysisId).foreach { _ =>
          fetchAnalyses() // Refresh the list after starting the analysis
        } recover { case ex => 
          setError(Some(s"Failed to run analysis: ${ex.getMessage}"))
        }
      }
      
      // Handle search term change
      val handleSearchChange = (e: ReactEventFrom[dom.html.Input]) => {
        setSearchTerm(e.target.value)
      }
      
      // Handle status filter change
      val handleStatusFilterChange = (e: ReactEventFrom[dom.html.Select]) => {
        setStatusFilter(e.target.value)
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
      
      // Get status display information
      def getStatusDisplay(status: String): (String, String) = status match {
        case Analysis.Completed => ("Completed", "bg-green-100 text-green-800")
        case Analysis.Running => ("Running", "bg-blue-100 text-blue-800")
        case Analysis.Pending => ("Pending", "bg-yellow-100 text-yellow-800")
        case Analysis.Failed => ("Failed", "bg-red-100 text-red-800")
        case _ => ("Unknown", "bg-gray-100 text-gray-800")
      }
      
      // Get analysis type display name
      def getAnalysisTypeDisplay(analysisType: String): String = analysisType match {
        case Analysis.KaplanMeier => "Kaplan-Meier"
        case Analysis.CoxRegression => "Cox Regression"
        case Analysis.RandomSurvivalForest => "Random Survival Forest"
        case _ => "Unknown"
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
            "Analyses"
          ),
          
          React.createElement(
            "button",
            js.Dynamic.literal(
              className = "px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition duration-150 flex items-center gap-2",
              onClick = () => handleCreateAnalysis()
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "span",
              null,
              "+"
            ),
            
            "Create Analysis"
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
                placeholder = "Search analyses...",
                className = "w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500",
                value = searchTerm,
                onChange = handleSearchChange
              ).asInstanceOf[js.Object]
            )
          ),
          
          // Status filter dropdown
          React.createElement(
            "select",
            js.Dynamic.literal(
              className = "px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500",
              value = statusFilter,
              onChange = handleStatusFilterChange
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "option",
              js.Dynamic.literal(
                value = "all"
              ).asInstanceOf[js.Object],
              "All Statuses"
            ),
            
            React.createElement(
              "option",
              js.Dynamic.literal(
                value = "completed"
              ).asInstanceOf[js.Object],
              "Completed"
            ),
            
            React.createElement(
              "option",
              js.Dynamic.literal(
                value = "running"
              ).asInstanceOf[js.Object],
              "Running"
            ),
            
            React.createElement(
              "option",
              js.Dynamic.literal(
                value = "pending"
              ).asInstanceOf[js.Object],
              "Pending"
            ),
            
            React.createElement(
              "option",
              js.Dynamic.literal(
                value = "failed"
              ).asInstanceOf[js.Object],
              "Failed"
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
            "Loading analyses..."
          )
        } else if (filteredAnalyses.isEmpty) {
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
              "No analyses found"
            ),
            
            React.createElement(
              "button",
              js.Dynamic.literal(
                className = "px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition duration-150",
                onClick = () => handleCreateAnalysis()
              ).asInstanceOf[js.Object],
              "Create your first analysis"
            )
          )
        } else {
          // Analysis list
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-white shadow rounded-lg overflow-hidden"
            ).asInstanceOf[js.Object],
            
            // Table
            React.createElement(
              "table",
              js.Dynamic.literal(
                className = "min-w-full divide-y divide-gray-200"
              ).asInstanceOf[js.Object],
              
              // Table header
              React.createElement(
                "thead",
                js.Dynamic.literal(
                  className = "bg-gray-50"
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "tr",
                  null,
                  
                  // Name column
                  React.createElement(
                    "th",
                    js.Dynamic.literal(
                      scope = "col",
                      className = "px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    ).asInstanceOf[js.Object],
                    "Name"
                  ),
                  
                  // Type column
                  React.createElement(
                    "th",
                    js.Dynamic.literal(
                      scope = "col",
                      className = "px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    ).asInstanceOf[js.Object],
                    "Type"
                  ),
                  
                  // Status column
                  React.createElement(
                    "th",
                    js.Dynamic.literal(
                      scope = "col",
                      className = "px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    ).asInstanceOf[js.Object],
                    "Status"
                  ),
                  
                  // Date column
                  React.createElement(
                    "th",
                    js.Dynamic.literal(
                      scope = "col",
                      className = "px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    ).asInstanceOf[js.Object],
                    "Created"
                  ),
                  
                  // Actions column
                  React.createElement(
                    "th",
                    js.Dynamic.literal(
                      scope = "col",
                      className = "px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider"
                    ).asInstanceOf[js.Object],
                    "Actions"
                  )
                )
              ),
              
              // Table body
              React.createElement(
                "tbody",
                js.Dynamic.literal(
                  className = "bg-white divide-y divide-gray-200"
                ).asInstanceOf[js.Object],
                
                filteredAnalyses.map { analysis =>
                  val (statusLabel, statusClass) = getStatusDisplay(analysis.status)
                  
                  React.createElement(
                    "tr",
                    js.Dynamic.literal(
                      key = analysis.id,
                      className = "hover:bg-gray-50 cursor-pointer",
                      onClick = () => handleSelectAnalysis(analysis.id)
                    ).asInstanceOf[js.Object],
                    
                    // Name cell
                    React.createElement(
                      "td",
                      js.Dynamic.literal(
                        className = "px-6 py-4 whitespace-nowrap"
                      ).asInstanceOf[js.Object],
                      
                      React.createElement(
                        "div",
                        js.Dynamic.literal(
                          className = "text-sm font-medium text-gray-900"
                        ).asInstanceOf[js.Object],
                        analysis.name
                      ),
                      
                      analysis.description.map { desc =>
                        React.createElement(
                          "div",
                          js.Dynamic.literal(
                            className = "text-sm text-gray-500 truncate max-w-xs"
                          ).asInstanceOf[js.Object],
                          desc
                        )
                      }.getOrElse(null)
                    ),
                    
                    // Type cell
                    React.createElement(
                      "td",
                      js.Dynamic.literal(
                        className = "px-6 py-4 whitespace-nowrap"
                      ).asInstanceOf[js.Object],
                      
                      React.createElement(
                        "div",
                        js.Dynamic.literal(
                          className = "text-sm text-gray-900"
                        ).asInstanceOf[js.Object],
                        getAnalysisTypeDisplay(analysis.analysisType)
                      )
                    ),
                    
                    // Status cell
                    React.createElement(
                      "td",
                      js.Dynamic.literal(
                        className = "px-6 py-4 whitespace-nowrap"
                      ).asInstanceOf[js.Object],
                      
                      React.createElement(
                        "span",
                        js.Dynamic.literal(
                          className = s"px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${statusClass}"
                        ).asInstanceOf[js.Object],
                        statusLabel
                      )
                    ),
                    
                    // Date cell
                    React.createElement(
                      "td",
                      js.Dynamic.literal(
                        className = "px-6 py-4 whitespace-nowrap text-sm text-gray-500"
                      ).asInstanceOf[js.Object],
                      formatDate(analysis.createdAt)
                    ),
                    
                    // Actions cell
                    React.createElement(
                      "td",
                      js.Dynamic.literal(
                        className = "px-6 py-4 whitespace-nowrap text-right text-sm font-medium",
                        onClick = (e: js.Dynamic) => {
                          e.stopPropagation()
                          (): Unit
                        }
                      ).asInstanceOf[js.Object],
                      
                      React.createElement(
                        "div",
                        js.Dynamic.literal(
                          className = "flex justify-end gap-2"
                        ).asInstanceOf[js.Object],
                        
                        // View button
                        React.createElement(
                          "button",
                          js.Dynamic.literal(
                            className = "text-blue-600 hover:text-blue-900",
                            title = "View details",
                            onClick = (e: js.Dynamic) => {
                              e.stopPropagation()
                              handleSelectAnalysis(analysis.id)
                            }
                          ).asInstanceOf[js.Object],
                          "ud83dudc41ufe0f"
                        ),
                        
                        // Run button (only show for non-running analyses)
                        if (analysis.status != Analysis.Running) {
                          React.createElement(
                            "button",
                            js.Dynamic.literal(
                              className = "text-green-600 hover:text-green-900",
                              title = "Run analysis",
                              onClick = (e: js.Dynamic) => {
                                e.stopPropagation()
                                handleRunAnalysis(analysis.id)
                              }
                            ).asInstanceOf[js.Object],
                            "u25b6ufe0f"
                          )
                        } else null,
                        
                        // Delete button
                        React.createElement(
                          "button",
                          js.Dynamic.literal(
                            className = "text-red-600 hover:text-red-900",
                            title = "Delete",
                            onClick = (e: js.Dynamic) => {
                              e.stopPropagation()
                              handleDeleteAnalysis(analysis.id)
                            }
                          ).asInstanceOf[js.Object],
                          "ud83duddd1ufe0f"
                        )
                      )
                    )
                  )
                }.toSeq: _*
              )
            )
          )
        }
      )
      
      // Render with layout
      Layout.render(
        Layout.Props(
          children = content,
          title = "Analyses",
          subtitle = "Create and manage survival analyses"
        )
      )
    }
  }
}

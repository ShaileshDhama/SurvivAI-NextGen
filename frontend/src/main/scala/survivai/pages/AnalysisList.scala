package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Success, Failure}
import survivai.bindings.ReactBindings.*
import survivai.components.*
import survivai.services.{AnalysisService, DatasetService}
import survivai.models.{Analysis, Dataset}
import survivai.contexts.LayoutContext

object AnalysisList {
  def render(): Element = {
    FC {
      // Use layout context to set page title
      val layoutContext = LayoutContext.useLayout()
      layoutContext.setTitle("Analyses")
      
      // States
      val analysesState = React.useState[js.Array[Analysis.Analysis]](js.Array())
      val analyses = analysesState(0).asInstanceOf[js.Array[Analysis.Analysis]]
      val setAnalyses = analysesState(1).asInstanceOf[js.Function1[js.Array[Analysis.Analysis], Unit]]
      
      val datasetsState = React.useState[Map[String, Dataset.Dataset]](Map())
      val datasets = datasetsState(0).asInstanceOf[Map[String, Dataset.Dataset]]
      val setDatasets = datasetsState(1).asInstanceOf[js.Function1[Map[String, Dataset.Dataset], Unit]]
      
      val loadingState = React.useState[Boolean](true)
      val loading = loadingState(0).asInstanceOf[Boolean]
      val setLoading = loadingState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      val errorState = React.useState[Option[String]](None)
      val error = errorState(0).asInstanceOf[Option[String]]
      val setError = errorState(1).asInstanceOf[js.Function1[Option[String], Unit]]
      
      val deleteModalState = React.useState[Option[Analysis.Analysis]](None)
      val deleteModal = deleteModalState(0).asInstanceOf[Option[Analysis.Analysis]]
      val setDeleteModal = deleteModalState(1).asInstanceOf[js.Function1[Option[Analysis.Analysis], Unit]]
      
      val deletingState = React.useState[Boolean](false)
      val deleting = deletingState(0).asInstanceOf[Boolean]
      val setDeleting = deletingState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      // Fetch analyses on component mount
      React.useEffect(() => {
        loadAnalyses()
        () => ()
      }, js.Array())
      
      // Function to load analyses
      def loadAnalyses(): Unit = {
        setLoading(true)
        setError(None)
        
        // First, fetch all analyses
        AnalysisService.getAnalyses(None).onComplete {
          case Success(analyses) =>
            setAnalyses(analyses.toArray[Analysis.Analysis])
            
            // Then fetch all associated datasets
            val datasetIds = analyses.map(_.datasetId).distinct
            
            // Only fetch datasets if there are analyses
            if (datasetIds.nonEmpty) {
              DatasetService.getDatasets(Some(datasetIds)).onComplete {
                case Success(datasets) =>
                  // Convert to map for easy lookup
                  val datasetMap = datasets.map(d => d.id -> d).toMap
                  setDatasets(datasetMap)
                  setLoading(false)
                  
                case Failure(exception) =>
                  setError(Some(s"Error loading datasets: ${exception.getMessage}"))
                  setLoading(false)
              }
            } else {
              setLoading(false)
            }
            
          case Failure(exception) =>
            setError(Some(s"Error loading analyses: ${exception.getMessage}"))
            setLoading(false)
        }
      }
      
      // Function to handle analysis deletion
      val handleDeleteAnalysis = js.Function1 { (analysis: Analysis.Analysis) =>
        setDeleteModal(Some(analysis))
      }
      
      // Function to confirm deletion
      val confirmDelete = js.Function0 { () =>
        deleteModal.foreach { analysis =>
          setDeleting(true)
          
          AnalysisService.deleteAnalysis(analysis.id).onComplete {
            case Success(_) =>
              // Remove deleted analysis from state
              setAnalyses(analyses.filter(_.id != analysis.id))
              setDeleteModal(None)
              setDeleting(false)
            case Failure(exception) =>
              setError(Some(s"Error deleting analysis: ${exception.getMessage}"))
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
      
      // Function to get status color class
      def getStatusColorClass(status: Analysis.Status): String = status match {
        case Analysis.Created => "bg-gray-100 text-gray-800"
        case Analysis.Running => "bg-blue-100 text-blue-800"
        case Analysis.Completed => "bg-green-100 text-green-800"
        case Analysis.Failed => "bg-red-100 text-red-800"
      }
      
      // Render analyses list page
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
            "Analyses"
          ),
          
          React.createElement(
            "a",
            js.Dynamic.literal(
              href = "#/analyses/new",
              className = "bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md inline-flex items-center transition-colors"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "span",
              js.Dynamic.literal(
                className = "mr-2"
              ).asInstanceOf[js.Object],
              "ud83dudd0d"
            ),
            
            "Create New Analysis"
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
                onClick = js.Function0(() => loadAnalyses()),
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
            "Loading analyses..."
          )
        } 
        // Empty state
        else if (analyses.isEmpty) {
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
              "ud83dudd0d"
            ),
            
            React.createElement(
              "h3",
              js.Dynamic.literal(
                className = "text-lg font-medium mb-2"
              ).asInstanceOf[js.Object],
              "No Analyses Found"
            ),
            
            React.createElement(
              "p",
              js.Dynamic.literal(
                className = "text-gray-500 mb-6"
              ).asInstanceOf[js.Object],
              "Create your first analysis to start exploring survival patterns in your data."
            ),
            
            React.createElement(
              "a",
              js.Dynamic.literal(
                href = "#/analyses/new",
                className = "bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-md inline-flex items-center transition-colors"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "span",
                js.Dynamic.literal(
                  className = "mr-2"
                ).asInstanceOf[js.Object],
                "ud83dudd0d"
              ),
              
              "Create New Analysis"
            )
          )
        } 
        // Analysis list
        else {
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-white rounded-lg shadow overflow-hidden"
            ).asInstanceOf[js.Object],
            
            // Analysis table
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
                  
                  // Dataset column
                  React.createElement(
                    "th",
                    js.Dynamic.literal(
                      scope = "col",
                      className = "px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    ).asInstanceOf[js.Object],
                    "Dataset"
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
                  
                  // Created column
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
                
                analyses.map { analysis =>
                  React.createElement(
                    "tr",
                    js.Dynamic.literal(
                      key = analysis.id,
                      className = "hover:bg-gray-50"
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
                          className = "flex items-center"
                        ).asInstanceOf[js.Object],
                        
                        React.createElement(
                          "div",
                          js.Dynamic.literal(
                            className = "text-sm font-medium text-gray-900"
                          ).asInstanceOf[js.Object],
                          analysis.name
                        )
                      )
                    ),
                    
                    // Dataset cell
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
                        datasets.get(analysis.datasetId).map(_.name).getOrElse(analysis.datasetId)
                      )
                    ),
                    
                    // Type cell
                    React.createElement(
                      "td",
                      js.Dynamic.literal(
                        className = "px-6 py-4 whitespace-nowrap text-sm text-gray-500"
                      ).asInstanceOf[js.Object],
                      Analysis.AnalysisType.toString(analysis.analysisType)
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
                          className = s"px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getStatusColorClass(analysis.status)}"
                        ).asInstanceOf[js.Object],
                        Analysis.Status.toString(analysis.status)
                      )
                    ),
                    
                    // Created cell
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
                        className = "px-6 py-4 whitespace-nowrap text-right text-sm font-medium"
                      ).asInstanceOf[js.Object],
                      
                      React.createElement(
                        "div",
                        js.Dynamic.literal(
                          className = "flex justify-end space-x-2"
                        ).asInstanceOf[js.Object],
                        
                        // View button
                        React.createElement(
                          "a",
                          js.Dynamic.literal(
                            href = s"#/analyses/${analysis.id}",
                            className = "text-blue-600 hover:text-blue-900"
                          ).asInstanceOf[js.Object],
                          "View"
                        ),
                        
                        // Delete button
                        React.createElement(
                          "button",
                          js.Dynamic.literal(
                            onClick = js.Function0(() => handleDeleteAnalysis(analysis)),
                            className = "text-red-600 hover:text-red-900"
                          ).asInstanceOf[js.Object],
                          "Delete"
                        )
                      )
                    )
                  )
                }: _*
              )
            )
          )
        },
        
        // Delete confirmation modal
        deleteModal.map { analysis =>
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
                s"Are you sure you want to delete the analysis '${analysis.name}'? This action cannot be undone."
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

package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Success, Failure}
import survivai.bindings.ReactBindings.*
import survivai.components.*
import survivai.services.DatasetService
import survivai.models.Dataset
import survivai.contexts.LayoutContext

object DatasetManagement {
  def render(): Element = {
    FC {
      // Use layout context to set page title
      val layoutContext = LayoutContext.useLayout()
      layoutContext.setTitle("Dataset Management")
      
      // State for datasets
      val datasetsState = React.useState[js.Array[Dataset.Dataset]](js.Array())
      val datasets = datasetsState(0).asInstanceOf[js.Array[Dataset.Dataset]]
      val setDatasets = datasetsState(1).asInstanceOf[js.Function1[js.Array[Dataset.Dataset], Unit]]
      
      // Loading and error states
      val loadingState = React.useState[Boolean](true)
      val loading = loadingState(0).asInstanceOf[Boolean]
      val setLoading = loadingState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      val errorState = React.useState[Option[String]](None)
      val error = errorState(0).asInstanceOf[Option[String]]
      val setError = errorState(1).asInstanceOf[js.Function1[Option[String], Unit]]
      
      // Delete confirmation modal state
      val deleteModalState = React.useState[Option[Dataset.Dataset]](None)
      val deleteModal = deleteModalState(0).asInstanceOf[Option[Dataset.Dataset]]
      val setDeleteModal = deleteModalState(1).asInstanceOf[js.Function1[Option[Dataset.Dataset], Unit]]
      
      val deletingState = React.useState[Boolean](false)
      val deleting = deletingState(0).asInstanceOf[Boolean]
      val setDeleting = deletingState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      // Fetch datasets on component mount
      React.useEffect(() => {
        loadDatasets()
        () => ()
      }, js.Array())
      
      // Function to load datasets
      def loadDatasets(): Unit = {
        setLoading(true)
        setError(None)
        
        DatasetService.getDatasets(None).onComplete {
          case Success(data) =>
            setDatasets(data.toArray[Dataset.Dataset])
            setLoading(false)
          case Failure(exception) =>
            setError(Some(s"Error loading datasets: ${exception.getMessage}"))
            setLoading(false)
        }
      }
      
      // Function to handle dataset deletion
      val handleDeleteDataset = js.Function1 { (dataset: Dataset.Dataset) =>
        setDeleteModal(Some(dataset))
      }
      
      // Function to confirm deletion
      val confirmDelete = js.Function0 { () =>
        deleteModal.foreach { dataset =>
          setDeleting(true)
          
          DatasetService.deleteDataset(dataset.id).onComplete {
            case Success(_) =>
              // Remove deleted dataset from state
              setDatasets(datasets.filter(_.id != dataset.id))
              setDeleteModal(None)
              setDeleting(false)
            case Failure(exception) =>
              setError(Some(s"Error deleting dataset: ${exception.getMessage}"))
              setDeleteModal(None)
              setDeleting(false)
          }
        }
      }
      
      // Function to cancel deletion
      val cancelDelete = js.Function0 { () =>
        setDeleteModal(None)
      }
      
      // Format file size
      def formatSize(bytes: Double): String = {
        if (bytes < 1024) {
          f"$bytes%.0f B"
        } else if (bytes < 1024 * 1024) {
          f"${bytes / 1024}%.1f KB"
        } else if (bytes < 1024 * 1024 * 1024) {
          f"${bytes / (1024 * 1024)}%.1f MB"
        } else {
          f"${bytes / (1024 * 1024 * 1024)}%.1f GB"
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
      
      // Render dataset management page
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = "space-y-6"
        ).asInstanceOf[js.Object],
        
        // Header with upload button
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
            "Dataset Management"
          ),
          
          React.createElement(
            "a",
            js.Dynamic.literal(
              href = "#/datasets/new",
              className = "bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md inline-flex items-center transition-colors"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "span",
              js.Dynamic.literal(
                className = "mr-2"
              ).asInstanceOf[js.Object],
              "ud83dudcbe"
            ),
            
            "Upload New Dataset"
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
                onClick = js.Function0(() => loadDatasets()),
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
            "Loading datasets..."
          )
        } 
        // Empty state
        else if (datasets.isEmpty) {
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
              "ud83dudcca"
            ),
            
            React.createElement(
              "h3",
              js.Dynamic.literal(
                className = "text-lg font-medium mb-2"
              ).asInstanceOf[js.Object],
              "No Datasets Found"
            ),
            
            React.createElement(
              "p",
              js.Dynamic.literal(
                className = "text-gray-500 mb-6"
              ).asInstanceOf[js.Object],
              "Upload your first dataset to get started with survival analysis."
            ),
            
            React.createElement(
              "a",
              js.Dynamic.literal(
                href = "#/datasets/new",
                className = "bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-md inline-flex items-center transition-colors"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "span",
                js.Dynamic.literal(
                  className = "mr-2"
                ).asInstanceOf[js.Object],
                "ud83dudcbe"
              ),
              
              "Upload New Dataset"
            )
          )
        } 
        // Dataset list
        else {
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-white rounded-lg shadow overflow-hidden"
            ).asInstanceOf[js.Object],
            
            // Dataset table
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
                  
                  // Description column
                  React.createElement(
                    "th",
                    js.Dynamic.literal(
                      scope = "col",
                      className = "px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    ).asInstanceOf[js.Object],
                    "Description"
                  ),
                  
                  // Rows column
                  React.createElement(
                    "th",
                    js.Dynamic.literal(
                      scope = "col",
                      className = "px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    ).asInstanceOf[js.Object],
                    "Rows"
                  ),
                  
                  // Size column
                  React.createElement(
                    "th",
                    js.Dynamic.literal(
                      scope = "col",
                      className = "px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    ).asInstanceOf[js.Object],
                    "Size"
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
                
                datasets.map { dataset =>
                  React.createElement(
                    "tr",
                    js.Dynamic.literal(
                      key = dataset.id,
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
                          dataset.name
                        )
                      )
                    ),
                    
                    // Description cell
                    React.createElement(
                      "td",
                      js.Dynamic.literal(
                        className = "px-6 py-4 whitespace-nowrap"
                      ).asInstanceOf[js.Object],
                      
                      React.createElement(
                        "div",
                        js.Dynamic.literal(
                          className = "text-sm text-gray-500"
                        ).asInstanceOf[js.Object],
                        dataset.description.getOrElse("-")
                      )
                    ),
                    
                    // Rows cell
                    React.createElement(
                      "td",
                      js.Dynamic.literal(
                        className = "px-6 py-4 whitespace-nowrap text-sm text-gray-500"
                      ).asInstanceOf[js.Object],
                      dataset.rowCount.toString
                    ),
                    
                    // Size cell
                    React.createElement(
                      "td",
                      js.Dynamic.literal(
                        className = "px-6 py-4 whitespace-nowrap text-sm text-gray-500"
                      ).asInstanceOf[js.Object],
                      formatSize(dataset.sizeBytes)
                    ),
                    
                    // Created cell
                    React.createElement(
                      "td",
                      js.Dynamic.literal(
                        className = "px-6 py-4 whitespace-nowrap text-sm text-gray-500"
                      ).asInstanceOf[js.Object],
                      formatDate(dataset.createdAt)
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
                            href = s"#/datasets/${dataset.id}",
                            className = "text-blue-600 hover:text-blue-900"
                          ).asInstanceOf[js.Object],
                          "View"
                        ),
                        
                        // Delete button
                        React.createElement(
                          "button",
                          js.Dynamic.literal(
                            onClick = js.Function0(() => handleDeleteDataset(dataset)),
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
        deleteModal.map { dataset =>
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
                s"Are you sure you want to delete the dataset '${dataset.name}'? This action cannot be undone."
              ),
              
              // Warning
              React.createElement(
                "p",
                js.Dynamic.literal(
                  className = "mb-6 text-orange-600 text-sm"
                ).asInstanceOf[js.Object],
                "Warning: All analyses using this dataset will also be deleted."
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

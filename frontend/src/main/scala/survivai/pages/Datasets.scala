package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import survivai.bindings.ReactBindings.*
import survivai.components.layout.Layout
import survivai.services.DatasetService
import survivai.models.Dataset
import java.util.Date
import org.scalajs.dom

object Datasets {
  def render(): Element = {
    FC {
      // State hooks
      val (datasets, setDatasets) = useState(js.Array[Dataset.Dataset]())
      val (isLoading, setIsLoading) = useState(true)
      val (searchTerm, setSearchTerm) = useState("")
      val (formatFilter, setFormatFilter) = useState("all")
      val (error, setError) = useState[Option[String]](None)
      
      // Fetch datasets on component mount
      useEffect(() => {
        fetchDatasets()
        () => ()
      }, js.Array())
      
      // Fetch datasets function
      def fetchDatasets(): Unit = {
        setIsLoading(true)
        setError(None)
        
        DatasetService.getDatasets(None).foreach { result =>
          setDatasets(js.Array(result: _*))
          setIsLoading(false)
        } recover { case ex => 
          console.error("Failed to fetch datasets:", ex.getMessage)
          setError(Some(s"Failed to load datasets: ${ex.getMessage}"))
          setIsLoading(false)
        }
      }
      
      // Filter datasets based on search term and format filter
      val filteredDatasets = datasets.filter { dataset =>
        val matchesSearch = searchTerm.isEmpty || 
          dataset.name.toLowerCase.contains(searchTerm.toLowerCase) || 
          dataset.description.exists(_.toLowerCase.contains(searchTerm.toLowerCase))
        
        val matchesFilter = formatFilter match {
          case "all" => true
          case "csv" => dataset.fileFormat.toLowerCase == "csv"
          case "excel" => dataset.fileFormat.toLowerCase == "xlsx" || dataset.fileFormat.toLowerCase == "xls"
          case _ => true
        }
        
        matchesSearch && matchesFilter
      }
      
      // Format file size
      def formatFileSize(bytes: Double): String = {
        if (bytes < 1024) {
          s"${bytes.toInt} B"
        } else if (bytes < 1024 * 1024) {
          f"${bytes / 1024}%.1f KB"
        } else if (bytes < 1024 * 1024 * 1024) {
          f"${bytes / (1024 * 1024)}%.1f MB"
        } else {
          f"${bytes / (1024 * 1024 * 1024)}%.1f GB"
        }
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
      
      // Handle dataset selection
      val handleSelectDataset = (datasetId: String) => {
        dom.window.location.href = s"#/datasets/${datasetId}"
      }
      
      // Handle upload dataset
      val handleUploadDataset = () => {
        dom.window.location.href = "#/datasets/new"
      }
      
      // Handle delete dataset
      val handleDeleteDataset = (datasetId: String) => {
        if (dom.window.confirm("Are you sure you want to delete this dataset? This action cannot be undone and will also delete any associated analyses.")) {
          DatasetService.deleteDataset(datasetId).foreach { success =>
            if (success) {
              fetchDatasets()
            } else {
              setError(Some("Failed to delete dataset"))
            }
          }
        }
      }
      
      // Handle search term change
      val handleSearchChange = (e: ReactEventFrom[dom.html.Input]) => {
        setSearchTerm(e.target.value)
      }
      
      // Handle format filter change
      val handleFormatFilterChange = (e: ReactEventFrom[dom.html.Select]) => {
        setFormatFilter(e.target.value)
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
            "Datasets"
          ),
          
          React.createElement(
            "button",
            js.Dynamic.literal(
              className = "px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition duration-150 flex items-center gap-2",
              onClick = () => handleUploadDataset()
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "span",
              null,
              "+"
            ),
            
            "Upload Dataset"
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
                placeholder = "Search datasets...",
                className = "w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500",
                value = searchTerm,
                onChange = handleSearchChange
              ).asInstanceOf[js.Object]
            )
          ),
          
          // Format filter dropdown
          React.createElement(
            "select",
            js.Dynamic.literal(
              className = "px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500",
              value = formatFilter,
              onChange = handleFormatFilterChange
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "option",
              js.Dynamic.literal(
                value = "all"
              ).asInstanceOf[js.Object],
              "All Formats"
            ),
            
            React.createElement(
              "option",
              js.Dynamic.literal(
                value = "csv"
              ).asInstanceOf[js.Object],
              "CSV Files"
            ),
            
            React.createElement(
              "option",
              js.Dynamic.literal(
                value = "excel"
              ).asInstanceOf[js.Object],
              "Excel Files"
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
            "Loading datasets..."
          )
        } else if (filteredDatasets.isEmpty) {
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
              "No datasets found"
            ),
            
            React.createElement(
              "button",
              js.Dynamic.literal(
                className = "px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition duration-150",
                onClick = () => handleUploadDataset()
              ).asInstanceOf[js.Object],
              "Upload your first dataset"
            )
          )
        } else {
          // Dataset list
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
                  
                  // Format column
                  React.createElement(
                    "th",
                    js.Dynamic.literal(
                      scope = "col",
                      className = "px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    ).asInstanceOf[js.Object],
                    "Format"
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
                  
                  // Date column
                  React.createElement(
                    "th",
                    js.Dynamic.literal(
                      scope = "col",
                      className = "px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    ).asInstanceOf[js.Object],
                    "Uploaded"
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
                
                filteredDatasets.map { dataset =>
                  React.createElement(
                    "tr",
                    js.Dynamic.literal(
                      key = dataset.id,
                      className = "hover:bg-gray-50 cursor-pointer",
                      onClick = () => handleSelectDataset(dataset.id)
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
                        dataset.name
                      ),
                      
                      dataset.description.map { desc =>
                        React.createElement(
                          "div",
                          js.Dynamic.literal(
                            className = "text-sm text-gray-500 truncate max-w-xs"
                          ).asInstanceOf[js.Object],
                          desc
                        )
                      }.getOrElse(null)
                    ),
                    
                    // Format cell
                    React.createElement(
                      "td",
                      js.Dynamic.literal(
                        className = "px-6 py-4 whitespace-nowrap"
                      ).asInstanceOf[js.Object],
                      
                      React.createElement(
                        "span",
                        js.Dynamic.literal(
                          className = "px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800"
                        ).asInstanceOf[js.Object],
                        dataset.fileFormat.toUpperCase()
                      )
                    ),
                    
                    // Size cell
                    React.createElement(
                      "td",
                      js.Dynamic.literal(
                        className = "px-6 py-4 whitespace-nowrap text-sm text-gray-500"
                      ).asInstanceOf[js.Object],
                      formatFileSize(dataset.fileSize)
                    ),
                    
                    // Date cell
                    React.createElement(
                      "td",
                      js.Dynamic.literal(
                        className = "px-6 py-4 whitespace-nowrap text-sm text-gray-500"
                      ).asInstanceOf[js.Object],
                      formatDate(dataset.createdAt)
                    ),
                    
                    // Rows cell
                    React.createElement(
                      "td",
                      js.Dynamic.literal(
                        className = "px-6 py-4 whitespace-nowrap text-sm text-gray-500"
                      ).asInstanceOf[js.Object],
                      dataset.rows.toString
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
                              handleSelectDataset(dataset.id)
                            }
                          ).asInstanceOf[js.Object],
                          "🔍"
                        ),
                        
                        // Delete button
                        React.createElement(
                          "button",
                          js.Dynamic.literal(
                            className = "text-red-600 hover:text-red-900",
                            title = "Delete",
                            onClick = (e: js.Dynamic) => {
                              e.stopPropagation()
                              handleDeleteDataset(dataset.id)
                            }
                          ).asInstanceOf[js.Object],
                          "🗑️"
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
          title = "Datasets",
          subtitle = "Manage your survival analysis datasets"
        )
      )
    }
  }
}

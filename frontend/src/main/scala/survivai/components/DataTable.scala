package survivai.components

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*
import org.scalajs.dom

object DataTable {
  // Props for the DataTable component
  case class Props(
    data: js.Array[js.Object], 
    columns: js.Array[Column],
    pagination: Boolean = true,
    pageSize: Int = 10,
    striped: Boolean = true,
    searchable: Boolean = false,
    sortable: Boolean = true,
    emptyMessage: String = "No data available",
    className: Option[String] = None,
    containerClassName: Option[String] = None
  )
  
  // Column definition
  case class Column(
    id: String,
    header: String,
    accessor: js.Function1[js.Object, js.Any],
    cellRenderer: Option[js.Function1[js.Any, Element]] = None,
    sortable: Boolean = true,
    width: Option[String] = None,
    className: Option[String] = None,
    headerClassName: Option[String] = None
  )
  
  def render(props: Props): Element = {
    FC {
      // State hooks
      val (data, setData) = useState[js.Array[js.Object]](props.data)
      val (sortColumn, setSortColumn) = useState[Option[String]](None)
      val (sortDirection, setSortDirection) = useState[String]("asc")
      val (currentPage, setCurrentPage) = useState(1)
      val (searchTerm, setSearchTerm) = useState("")
      
      // Derived state for pagination
      val totalPages = Math.max(1, Math.ceil(data.length.toDouble / props.pageSize).toInt)
      val startIdx = (currentPage - 1) * props.pageSize
      val endIdx = Math.min(startIdx + props.pageSize, data.length)
      
      // Handle sort column change
      val handleSortChange = (columnId: String) => {
        if (!props.sortable) return
        
        if (sortColumn.contains(columnId)) {
          // Toggle sort direction if same column
          setSortDirection(if (sortDirection == "asc") "desc" else "asc")
        } else {
          // New sort column
          setSortColumn(Some(columnId))
          setSortDirection("asc")
        }
      }
      
      // Handle search term change
      val handleSearchChange = (e: ReactEventFrom[dom.html.Input]) => {
        val term = e.target.value
        setSearchTerm(term)
        setCurrentPage(1) // Reset to first page on search
      }
      
      // Effect to sort and filter data when dependencies change
      useEffect(() => {
        var filteredData = props.data
        
        // Apply search filter if searchable
        if (props.searchable && searchTerm.nonEmpty) {
          val lowerSearchTerm = searchTerm.toLowerCase
          filteredData = props.data.filter { row =>
            // Check if any value in the row contains the search term
            val rowValues = props.columns.map(_.accessor(row))
            rowValues.exists { value =>
              if (value == null) false
              else value.toString.toLowerCase.contains(lowerSearchTerm)
            }
          }
        }
        
        // Apply sorting if sortable and a column is selected
        if (props.sortable && sortColumn.nonEmpty) {
          val columnId = sortColumn.get
          val column = props.columns.find(_.id == columnId).get
          
          // Sort the data based on column accessor
          filteredData = js.Array(filteredData.sortWith { (a, b) =>
            val valueA = column.accessor(a)
            val valueB = column.accessor(b)
            
            val comparison = (valueA, valueB) match {
              case (a: String, b: String) => a.compareTo(b)
              case (a: Double, b: Double) => a.compareTo(b)
              case (a: Int, b: Int) => a.compareTo(b)
              case (a: Boolean, b: Boolean) => a.compareTo(b)
              case (a: js.Date, b: js.Date) => a.getTime().compareTo(b.getTime())
              case _ => valueA.toString.compareTo(valueB.toString)
            }
            
            if (sortDirection == "asc") comparison < 0 else comparison > 0
          }: _*)
        }
        
        setData(filteredData)
      }, js.Array(props.data, searchTerm, sortColumn, sortDirection))
      
      // Get current page data
      val currentPageData = data.slice(startIdx, endIdx)
      
      // Container class names
      val containerClass = s"overflow-hidden rounded-lg ${props.containerClassName.getOrElse("")}" 
      val tableClass = s"min-w-full divide-y divide-gray-200 ${props.className.getOrElse("")}" 
      
      // Render the component
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = containerClass
        ).asInstanceOf[js.Object],
        
        // Search input if searchable
        if (props.searchable) {
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "mb-4"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "input",
              js.Dynamic.literal(
                type = "text",
                value = searchTerm,
                onChange = handleSearchChange,
                placeholder = "Search...",
                className = "w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500"
              ).asInstanceOf[js.Object]
            )
          )
        } else null,
        
        // Table container with horizontal scroll
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "overflow-x-auto"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "table",
            js.Dynamic.literal(
              className = tableClass
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
                
                // Render column headers
                props.columns.map { column =>
                  val headerClass = s"px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider ${if (props.sortable && column.sortable) "cursor-pointer hover:bg-gray-100" else ""} ${column.headerClassName.getOrElse("")}"
                  val style = if (column.width.isDefined) js.Dynamic.literal(width = column.width.get) else null
                  
                  React.createElement(
                    "th",
                    js.Dynamic.literal(
                      key = column.id,
                      className = headerClass,
                      style = style,
                      onClick = () => if (props.sortable && column.sortable) handleSortChange(column.id)
                    ).asInstanceOf[js.Object],
                    
                    React.createElement(
                      "div",
                      js.Dynamic.literal(
                        className = "flex items-center gap-1"
                      ).asInstanceOf[js.Object],
                      
                      column.header,
                      
                      // Sort indicator
                      if (props.sortable && column.sortable && sortColumn.contains(column.id)) {
                        val iconPath = if (sortDirection == "asc") {
                          "M5 15l7-7 7 7"
                        } else {
                          "M19 9l-7 7-7-7"
                        }
                        
                        React.createElement(
                          "svg",
                          js.Dynamic.literal(
                            xmlns = "http://www.w3.org/2000/svg",
                            className = "h-4 w-4 ml-1",
                            fill = "none",
                            viewBox = "0 0 24 24",
                            stroke = "currentColor"
                          ).asInstanceOf[js.Object],
                          
                          React.createElement(
                            "path",
                            js.Dynamic.literal(
                              strokeLinecap = "round",
                              strokeLinejoin = "round",
                              strokeWidth = 2,
                              d = iconPath
                            ).asInstanceOf[js.Object]
                          )
                        )
                      } else null
                    )
                  )
                }.toArray
              )
            ),
            
            // Table body
            React.createElement(
              "tbody",
              js.Dynamic.literal(
                className = if (props.striped) "bg-white divide-y divide-gray-200" else "divide-y divide-gray-200"
              ).asInstanceOf[js.Object],
              
              if (currentPageData.length == 0) {
                // Empty state
                React.createElement(
                  "tr",
                  null,
                  
                  React.createElement(
                    "td",
                    js.Dynamic.literal(
                      colSpan = props.columns.length,
                      className = "px-6 py-4 text-center text-sm text-gray-500"
                    ).asInstanceOf[js.Object],
                    
                    props.emptyMessage
                  )
                )
              } else {
                // Data rows
                currentPageData.zipWithIndex.map { case (row, rowIndex) =>
                  val rowClassName = if (props.striped && rowIndex % 2 == 1) "bg-gray-50" else ""
                  
                  React.createElement(
                    "tr",
                    js.Dynamic.literal(
                      key = rowIndex.toString,
                      className = rowClassName
                    ).asInstanceOf[js.Object],
                    
                    // Render cells for each column
                    props.columns.map { column =>
                      val cellValue = column.accessor(row)
                      val cellClassName = s"px-6 py-4 text-sm text-gray-500 ${column.className.getOrElse("")}"
                      
                      React.createElement(
                        "td",
                        js.Dynamic.literal(
                          key = column.id,
                          className = cellClassName
                        ).asInstanceOf[js.Object],
                        
                        // Render cell with custom renderer if provided
                        column.cellRenderer match {
                          case Some(renderer) => renderer(cellValue)
                          case None => if (cellValue == null) "" else cellValue.toString
                        }
                      )
                    }.toArray
                  )
                }.toArray
              }
            )
          )
        ),
        
        // Pagination controls
        if (props.pagination && totalPages > 1) {
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "px-4 py-3 flex items-center justify-between border-t border-gray-200 sm:px-6"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "flex-1 flex justify-between sm:hidden"
              ).asInstanceOf[js.Object],
              
              // Mobile pagination
              React.createElement(
                "button",
                js.Dynamic.literal(
                  onClick = () => setCurrentPage(Math.max(1, currentPage - 1)),
                  disabled = currentPage == 1,
                  className = "relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                ).asInstanceOf[js.Object],
                "Previous"
              ),
              
              React.createElement(
                "button",
                js.Dynamic.literal(
                  onClick = () => setCurrentPage(Math.min(totalPages, currentPage + 1)),
                  disabled = currentPage == totalPages,
                  className = "ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                ).asInstanceOf[js.Object],
                "Next"
              )
            ),
            
            // Desktop pagination
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "hidden sm:flex-1 sm:flex sm:items-center sm:justify-between"
              ).asInstanceOf[js.Object],
              
              // Results summary
              React.createElement(
                "div",
                null,
                
                React.createElement(
                  "p",
                  js.Dynamic.literal(
                    className = "text-sm text-gray-700"
                  ).asInstanceOf[js.Object],
                  
                  React.createElement(
                    "span",
                    null,
                    "Showing "
                  ),
                  
                  React.createElement(
                    "span",
                    js.Dynamic.literal(
                      className = "font-medium"
                    ).asInstanceOf[js.Object],
                    s"${startIdx + 1}"
                  ),
                  
                  React.createElement(
                    "span",
                    null,
                    " to "
                  ),
                  
                  React.createElement(
                    "span",
                    js.Dynamic.literal(
                      className = "font-medium"
                    ).asInstanceOf[js.Object],
                    s"${Math.min(endIdx, data.length)}"
                  ),
                  
                  React.createElement(
                    "span",
                    null,
                    " of "
                  ),
                  
                  React.createElement(
                    "span",
                    js.Dynamic.literal(
                      className = "font-medium"
                    ).asInstanceOf[js.Object],
                    s"${data.length}"
                  ),
                  
                  React.createElement(
                    "span",
                    null,
                    " results"
                  )
                )
              ),
              
              // Pagination buttons
              React.createElement(
                "div",
                null,
                
                React.createElement(
                  "nav",
                  js.Dynamic.literal(
                    className = "relative z-0 inline-flex rounded-md shadow-sm -space-x-px",
                    "aria-label" = "Pagination"
                  ).asInstanceOf[js.Object],
                  
                  // Previous page button
                  React.createElement(
                    "button",
                    js.Dynamic.literal(
                      onClick = () => setCurrentPage(Math.max(1, currentPage - 1)),
                      disabled = currentPage == 1,
                      className = "relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    ).asInstanceOf[js.Object],
                    
                    React.createElement(
                      "span",
                      js.Dynamic.literal(
                        className = "sr-only"
                      ).asInstanceOf[js.Object],
                      "Previous"
                    ),
                    
                    React.createElement(
                      "svg",
                      js.Dynamic.literal(
                        className = "h-5 w-5",
                        xmlns = "http://www.w3.org/2000/svg",
                        viewBox = "0 0 20 20",
                        fill = "currentColor",
                        "aria-hidden" = "true"
                      ).asInstanceOf[js.Object],
                      
                      React.createElement(
                        "path",
                        js.Dynamic.literal(
                          fillRule = "evenodd",
                          d = "M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z",
                          clipRule = "evenodd"
                        ).asInstanceOf[js.Object]
                      )
                    )
                  ),
                  
                  // Page number buttons
                  (1 to totalPages).map { page =>
                    val isActive = page == currentPage
                    val buttonClass = if (isActive) {
                      "z-10 bg-blue-50 border-blue-500 text-blue-600 relative inline-flex items-center px-4 py-2 border text-sm font-medium"
                    } else {
                      "bg-white border-gray-300 text-gray-500 hover:bg-gray-50 relative inline-flex items-center px-4 py-2 border text-sm font-medium"
                    }
                    
                    React.createElement(
                      "button",
                      js.Dynamic.literal(
                        key = page.toString,
                        onClick = () => setCurrentPage(page),
                        className = buttonClass,
                        "aria-current" = if (isActive) "page" else null
                      ).asInstanceOf[js.Object],
                      page.toString
                    )
                  }.toArray,
                  
                  // Next page button
                  React.createElement(
                    "button",
                    js.Dynamic.literal(
                      onClick = () => setCurrentPage(Math.min(totalPages, currentPage + 1)),
                      disabled = currentPage == totalPages,
                      className = "relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    ).asInstanceOf[js.Object],
                    
                    React.createElement(
                      "span",
                      js.Dynamic.literal(
                        className = "sr-only"
                      ).asInstanceOf[js.Object],
                      "Next"
                    ),
                    
                    React.createElement(
                      "svg",
                      js.Dynamic.literal(
                        className = "h-5 w-5",
                        xmlns = "http://www.w3.org/2000/svg",
                        viewBox = "0 0 20 20",
                        fill = "currentColor",
                        "aria-hidden" = "true"
                      ).asInstanceOf[js.Object],
                      
                      React.createElement(
                        "path",
                        js.Dynamic.literal(
                          fillRule = "evenodd",
                          d = "M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z",
                          clipRule = "evenodd"
                        ).asInstanceOf[js.Object]
                      )
                    )
                  )
                )
              )
            )
          )
        } else null
      )
    }
  }
  
  // Static methods to create column definitions
  object Column {
    // Helper to create a text column with default rendering
    def text(id: String, header: String, accessor: js.Function1[js.Object, js.Any], width: Option[String] = None): Column = {
      Column(
        id = id,
        header = header,
        accessor = accessor,
        width = width
      )
    }
    
    // Helper to create a date column with formatted date rendering
    def date(id: String, header: String, accessor: js.Function1[js.Object, js.Date], format: String = "medium", width: Option[String] = None): Column = {
      // Format options based on requested format
      val dateFormatOptions = format match {
        case "short" => js.Dynamic.literal(
          year = "numeric",
          month = "numeric",
          day = "numeric"
        )
        case "medium" => js.Dynamic.literal(
          year = "numeric",
          month = "short",
          day = "numeric"
        )
        case "long" => js.Dynamic.literal(
          year = "numeric",
          month = "long",
          day = "numeric",
          hour = "2-digit",
          minute = "2-digit"
        )
        case _ => js.Dynamic.literal(
          year = "numeric",
          month = "short",
          day = "numeric"
        )
      }
      
      // Custom renderer for date values
      val renderer = (value: js.Any) => {
        if (value == null || js.isUndefined(value)) {
          null
        } else {
          val dateValue = value.asInstanceOf[js.Date]
          try {
            val formattedDate = dateValue.toLocaleDateString("en-US", dateFormatOptions.asInstanceOf[js.Object])
            formattedDate
          } catch {
            case _: Throwable => "Invalid date"
          }
        }
      }
      
      Column(
        id = id,
        header = header,
        accessor = accessor,
        cellRenderer = Some(renderer),
        width = width
      )
    }
    
    // Helper to create a boolean column with Yes/No rendering
    def boolean(id: String, header: String, accessor: js.Function1[js.Object, Boolean], width: Option[String] = None): Column = {
      val renderer = (value: js.Any) => {
        if (value == null || js.isUndefined(value)) {
          null
        } else {
          val boolValue = value.asInstanceOf[Boolean]
          if (boolValue) "Yes" else "No"
        }
      }
      
      Column(
        id = id,
        header = header,
        accessor = accessor,
        cellRenderer = Some(renderer),
        width = width
      )
    }
    
    // Helper to create an actions column with custom buttons
    def actions(id: String, header: String, actions: js.Array[ActionsConfig], width: Option[String] = None): Column = {
      val renderer = (value: js.Any) => {
        val row = value.asInstanceOf[js.Object]
        
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "flex space-x-2"
          ).asInstanceOf[js.Object],
          
          actions.map { action =>
            // Determine if action should be shown
            val visible = action.isVisible match {
              case Some(fn) => fn(row)
              case None => true
            }
            
            if (!visible) null
            else {
              React.createElement(
                "button",
                js.Dynamic.literal(
                  key = action.label,
                  onClick = () => action.onClick(row),
                  className = action.className.getOrElse("inline-flex items-center px-2.5 py-1.5 border border-transparent text-xs font-medium rounded text-gray-700 bg-gray-100 hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"),
                  title = action.label
                ).asInstanceOf[js.Object],
                
                // If SVG icon is provided
                if (action.icon.isDefined) {
                  js.Array(
                    React.createElement(
                      "span",
                      js.Dynamic.literal(
                        className = "mr-1",
                        dangerouslySetInnerHTML = js.Dynamic.literal(
                          __html = action.icon.get
                        )
                      ).asInstanceOf[js.Object]
                    ),
                    
                    if (action.showLabel) action.label else null
                  )
                } else action.label
              )
            }
          }.filter(_ != null).toArray
        )
      }
      
      Column(
        id = id,
        header = header,
        accessor = row => row, // Pass the entire row
        cellRenderer = Some(renderer),
        sortable = false,
        width = width,
        className = Some("whitespace-nowrap")
      )
    }
  }
  
  // Configuration for action buttons
  case class ActionsConfig(
    label: String,
    onClick: js.Function1[js.Object, Unit],
    icon: Option[String] = None,
    showLabel: Boolean = true,
    className: Option[String] = None,
    isVisible: Option[js.Function1[js.Object, Boolean]] = None
  )
}

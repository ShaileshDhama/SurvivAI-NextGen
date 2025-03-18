package survivai.components.reports

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*
import org.scalajs.dom
import java.util.UUID

object ReportList {
  // Report data structure
  case class Report(
    id: String,
    title: String,
    description: Option[String],
    analysisId: String,
    createdAt: js.Date,
    updatedAt: Option[js.Date]
  )
  
  // Component props
  case class Props(
    reports: js.Array[Report],
    isLoading: Boolean = false,
    onViewReport: js.Function1[String, Unit],
    onDeleteReport: js.Function1[String, Unit],
    className: Option[String] = None
  )
  
  def render(props: Props): Element = {
    FC {
      // Format date helper
      val formatDate = (date: js.Date): String => {
        try {
          val options = js.Dynamic.literal(
            year = "numeric",
            month = "short",
            day = "numeric"
          )
          date.toLocaleDateString("en-US", options.asInstanceOf[js.Object])
        } catch {
          case _: Throwable => "Unknown date"
        }
      }
      
      // Container class
      val containerClass = s"overflow-hidden bg-white shadow-md rounded-lg ${props.className.getOrElse("")}"
      
      // Render component
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = containerClass
        ).asInstanceOf[js.Object],
        
        // Reports list header
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "px-4 py-5 border-b border-gray-200 sm:px-6"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "h3",
            js.Dynamic.literal(
              className = "text-lg leading-6 font-medium text-gray-900"
            ).asInstanceOf[js.Object],
            "Analysis Reports"
          ),
          
          React.createElement(
            "p",
            js.Dynamic.literal(
              className = "mt-1 max-w-2xl text-sm text-gray-500"
            ).asInstanceOf[js.Object],
            "View and manage your survival analysis reports"
          )
        ),
        
        // Loading state
        if (props.isLoading) {
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "p-6 text-center"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "inline-block animate-spin h-8 w-8 border-4 border-gray-300 rounded-full border-t-blue-600 mb-2"
              ).asInstanceOf[js.Object]
            ),
            
            React.createElement(
              "p",
              js.Dynamic.literal(
                className = "text-gray-600"
              ).asInstanceOf[js.Object],
              "Loading reports..."
            )
          )
        } 
        // Empty state
        else if (props.reports.length == 0) {
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "p-8 text-center"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "svg",
              js.Dynamic.literal(
                className = "mx-auto h-12 w-12 text-gray-400",
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
                  d = "M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                ).asInstanceOf[js.Object]
              )
            ),
            
            React.createElement(
              "h3",
              js.Dynamic.literal(
                className = "mt-2 text-sm font-medium text-gray-900"
              ).asInstanceOf[js.Object],
              "No reports"
            ),
            
            React.createElement(
              "p",
              js.Dynamic.literal(
                className = "mt-1 text-sm text-gray-500"
              ).asInstanceOf[js.Object],
              "Generate a report from one of your analyses to get started."
            )
          )
        } 
        // Reports list
        else {
          React.createElement(
            "ul",
            js.Dynamic.literal(
              className = "divide-y divide-gray-200"
            ).asInstanceOf[js.Object],
            
            props.reports.map { report =>
              React.createElement(
                "li",
                js.Dynamic.literal(
                  key = report.id,
                  className = "px-4 py-4 sm:px-6 hover:bg-gray-50 transition-colors"
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "flex items-center justify-between"
                  ).asInstanceOf[js.Object],
                  
                  // Report info
                  React.createElement(
                    "div",
                    js.Dynamic.literal(
                      className = "flex-1 min-w-0"
                    ).asInstanceOf[js.Object],
                    
                    // Title
                    React.createElement(
                      "h4",
                      js.Dynamic.literal(
                        className = "text-sm font-medium text-blue-600 truncate cursor-pointer hover:underline",
                        onClick = (_: ReactEventFrom[dom.html.Element]) => props.onViewReport(report.id)
                      ).asInstanceOf[js.Object],
                      report.title
                    ),
                    
                    // Description (if present)
                    report.description.map { desc =>
                      React.createElement(
                        "p",
                        js.Dynamic.literal(
                          className = "mt-1 text-sm text-gray-500 truncate"
                        ).asInstanceOf[js.Object],
                        desc
                      )
                    }.orNull,
                    
                    // Created date
                    React.createElement(
                      "p",
                      js.Dynamic.literal(
                        className = "mt-1 text-xs text-gray-500"
                      ).asInstanceOf[js.Object],
                      s"Created on ${formatDate(report.createdAt)}"
                    )
                  ),
                  
                  // Action buttons
                  React.createElement(
                    "div",
                    js.Dynamic.literal(
                      className = "ml-4 flex-shrink-0 flex space-x-2"
                    ).asInstanceOf[js.Object],
                    
                    // View button
                    React.createElement(
                      "button",
                      js.Dynamic.literal(
                        `type` = "button",
                        className = "bg-white text-blue-600 hover:text-blue-800 p-1 rounded",
                        onClick = (_: ReactEventFrom[dom.html.Element]) => props.onViewReport(report.id)
                      ).asInstanceOf[js.Object],
                      
                      React.createElement(
                        "svg",
                        js.Dynamic.literal(
                          className = "h-5 w-5",
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
                            d = "M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                          ).asInstanceOf[js.Object]
                        ),
                        
                        React.createElement(
                          "path",
                          js.Dynamic.literal(
                            strokeLinecap = "round",
                            strokeLinejoin = "round",
                            strokeWidth = 2,
                            d = "M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                          ).asInstanceOf[js.Object]
                        )
                      )
                    ),
                    
                    // Delete button
                    React.createElement(
                      "button",
                      js.Dynamic.literal(
                        `type` = "button",
                        className = "bg-white text-red-600 hover:text-red-800 p-1 rounded",
                        onClick = (_: ReactEventFrom[dom.html.Element]) => props.onDeleteReport(report.id)
                      ).asInstanceOf[js.Object],
                      
                      React.createElement(
                        "svg",
                        js.Dynamic.literal(
                          className = "h-5 w-5",
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
                            d = "M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                          ).asInstanceOf[js.Object]
                        )
                      )
                    )
                  )
                )
              )
            }
          )
        }
      )
    }
  }
}

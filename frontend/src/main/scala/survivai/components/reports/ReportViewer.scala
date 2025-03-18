package survivai.components.reports

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*
import org.scalajs.dom
import scala.util.{Success, Failure}

object ReportViewer {
  // Report data structure
  case class Report(
    id: String,
    title: String,
    description: Option[String],
    content: String,
    analysisId: String,
    createdAt: js.Date,
    updatedAt: Option[js.Date]
  )
  
  // Component props
  case class Props(
    reportId: String,
    onClose: js.Function0[Unit],
    className: Option[String] = None
  )
  
  def render(props: Props): Element = {
    FC {
      // State hooks
      val (report, setReport) = useState[Option[Report]](None)
      val (isLoading, setIsLoading) = useState(true)
      val (error, setError) = useState[Option[String]](None)
      val (showChatbot, setShowChatbot) = useState(false)
      
      // Effect to fetch report data
      useEffect(() => {
        setIsLoading(true)
        setError(None)
        
        dom.fetch(s"/api/v1/reports/${props.reportId}")
          .`then`[js.Dynamic](response => {
            if (!response.ok.asInstanceOf[Boolean]) {
              throw js.Error(s"Failed to fetch report: ${response.statusText.asInstanceOf[String]}")
            }
            response.json().asInstanceOf[js.Promise[js.Dynamic]]
          })
          .`then`[Unit](data => {
            val description = if (js.isUndefined(data.description) || data.description == null) {
              None
            } else {
              Some(data.description.asInstanceOf[String])
            }
            
            val updatedAt = if (js.isUndefined(data.updatedAt) || data.updatedAt == null) {
              None
            } else {
              Some(new js.Date(data.updatedAt.asInstanceOf[String]))
            }
            
            setReport(Some(Report(
              id = data.id.asInstanceOf[String],
              title = data.title.asInstanceOf[String],
              description = description,
              content = data.content.asInstanceOf[String],
              analysisId = data.analysisId.asInstanceOf[String],
              createdAt = new js.Date(data.createdAt.asInstanceOf[String]),
              updatedAt = updatedAt
            )))
            setIsLoading(false)
          })
          .`catch`(error => {
            console.error("Error fetching report:", error)
            setIsLoading(false)
            setError(Some(s"Failed to load report: ${error.toString()}"))
          })
        
        // No cleanup needed
        () => ()
      }, js.Array(props.reportId))
      
      // Format date helper
      val formatDate = (date: js.Date): String => {
        try {
          val options = js.Dynamic.literal(
            year = "numeric",
            month = "long",
            day = "numeric"
          )
          date.toLocaleDateString("en-US", options.asInstanceOf[js.Object])
        } catch {
          case _: Throwable => "Unknown date"
        }
      }
      
      // Convert markdown to HTML
      val convertMarkdownToHtml = (markdown: String): String => {
        // We would use a markdown library like marked.js here
        // For now, we'll use a simple placeholder approach
        val marked = js.Dynamic.global.window.marked
        
        if (!js.isUndefined(marked) && marked != null) {
          marked(markdown).asInstanceOf[String]
        } else {
          // Simple fallback if marked.js is not available
          markdown
            .replace("# ", "<h1>")
            .replace("## ", "<h2>")
            .replace("### ", "<h3>")
            .replace("\n\n", "<br/><br/>")
        }
      }
      
      // Toggle chatbot visibility
      val toggleChatbot = (_: ReactEventFrom[dom.html.Element]) => {
        setShowChatbot(prev => !prev)
      }
      
      // Handle print action
      val handlePrint = (_: ReactEventFrom[dom.html.Element]) => {
        dom.window.print()
      }
      
      // Container class
      val containerClass = s"bg-white shadow-lg rounded-lg overflow-hidden ${props.className.getOrElse("")}"
      
      // Render component
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = containerClass
        ).asInstanceOf[js.Object],
        
        // Report header
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "px-6 py-4 border-b border-gray-200 flex justify-between items-center print:hidden"
          ).asInstanceOf[js.Object],
          
          // Title section
          React.createElement(
            "div",
            js.Dynamic.literal().asInstanceOf[js.Object],
            
            React.createElement(
              "h2",
              js.Dynamic.literal(
                className = "text-xl font-semibold text-gray-800"
              ).asInstanceOf[js.Object],
              report.map(_.title).getOrElse("Loading report...")
            ),
            
            report.flatMap(_.description).map(desc => 
              React.createElement(
                "p",
                js.Dynamic.literal(
                  className = "text-sm text-gray-500 mt-1"
                ).asInstanceOf[js.Object],
                desc
              )
            ).orNull
          ),
          
          // Actions
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "flex space-x-2"
            ).asInstanceOf[js.Object],
            
            // Print button
            React.createElement(
              "button",
              js.Dynamic.literal(
                onClick = handlePrint,
                className = "p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-full",
                title = "Print report"
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
                    d = "M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z"
                  ).asInstanceOf[js.Object]
                )
              )
            ),
            
            // Chat button
            React.createElement(
              "button",
              js.Dynamic.literal(
                onClick = toggleChatbot,
                className = "p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-full",
                title = "Ask questions about this report"
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
                    d = "M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                  ).asInstanceOf[js.Object]
                )
              )
            ),
            
            // Close button
            React.createElement(
              "button",
              js.Dynamic.literal(
                onClick = props.onClose,
                className = "p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-full",
                title = "Close report"
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
                    d = "M6 18L18 6M6 6l12 12"
                  ).asInstanceOf[js.Object]
                )
              )
            )
          )
        ),
        
        // Main content area
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "flex flex-col md:flex-row"
          ).asInstanceOf[js.Object],
          
          // Report content
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "flex-1 p-6 overflow-auto"
            ).asInstanceOf[js.Object],
            
            // Loading state
            if (isLoading) {
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "flex justify-center items-center h-64"
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "inline-block animate-spin h-8 w-8 border-4 border-gray-300 rounded-full border-t-blue-600"
                  ).asInstanceOf[js.Object]
                )
              )
            } 
            // Error state
            else if (error.isDefined) {
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded"
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "p",
                  js.Dynamic.literal(
                    className = "font-medium"
                  ).asInstanceOf[js.Object],
                  "Error loading report"
                ),
                
                React.createElement(
                  "p",
                  js.Dynamic.literal(
                    className = "text-sm"
                  ).asInstanceOf[js.Object],
                  error.getOrElse("")
                )
              )
            } 
            // Report content
            else if (report.isDefined) {
              val r = report.get
              
              js.Array(
                // Report header for print
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    key = "print-header",
                    className = "hidden print:block mb-8"
                  ).asInstanceOf[js.Object],
                  
                  React.createElement(
                    "h1",
                    js.Dynamic.literal(
                      className = "text-2xl font-bold text-center"
                    ).asInstanceOf[js.Object],
                    r.title
                  ),
                  
                  r.description.map(desc => 
                    React.createElement(
                      "p",
                      js.Dynamic.literal(
                        className = "text-gray-600 text-center mt-2"
                      ).asInstanceOf[js.Object],
                      desc
                    )
                  ).orNull,
                  
                  React.createElement(
                    "p",
                    js.Dynamic.literal(
                      className = "text-gray-500 text-sm text-center mt-4"
                    ).asInstanceOf[js.Object],
                    s"Generated on ${formatDate(r.createdAt)}"
                  )
                ),
                
                // Actual report content
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    key = "report-content",
                    className = "prose prose-blue max-w-none report-content",
                    dangerouslySetInnerHTML = js.Dynamic.literal(
                      __html = convertMarkdownToHtml(r.content)
                    )
                  ).asInstanceOf[js.Object]
                )
              )
            } else null
          ),
          
          // Chatbot sidebar (if visible)
          if (showChatbot && report.isDefined) {
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "w-full md:w-96 border-t md:border-t-0 md:border-l border-gray-200 print:hidden"
              ).asInstanceOf[js.Object],
              
              ReportChatbot.render(ReportChatbot.Props(
                reportId = Some(props.reportId),
                modelId = report.get.analysisId,
                onClose = Some(() => setShowChatbot(false)),
                className = Some("h-[600px] md:h-auto")
              ))
            )
          } else null
        )
      )
    }
  }
}

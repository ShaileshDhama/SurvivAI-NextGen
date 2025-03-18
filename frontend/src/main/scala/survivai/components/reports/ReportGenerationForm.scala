package survivai.components.reports

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*
import org.scalajs.dom
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Success, Failure}
import survivai.models.Analysis

object ReportGenerationForm {
  case class Props(
    analysisId: String,
    onReportGenerated: js.Function1[String, Unit],
    onCancel: js.Function0[Unit],
    className: Option[String] = None
  )
  
  def render(props: Props): Element = {
    FC {
      // State hooks
      val (title, setTitle) = useState("")
      val (description, setDescription) = useState("")
      val (includeSections, setIncludeSections) = useState(js.Dictionary(
        "summary" -> true,
        "dataDescription" -> true,
        "methodology" -> true,
        "results" -> true,
        "visualizations" -> true,
        "interpretations" -> true,
        "recommendations" -> true
      ))
      val (isSubmitting, setIsSubmitting) = useState(false)
      val (error, setError) = useState(Option.empty[String])
      
      // Handle form submission
      val handleSubmit = (event: ReactEventFrom[dom.html.Form]) => {
        event.preventDefault()
        
        if (title.trim() == "") {
          setError(Some("Please enter a report title"))
          return
        }
        
        setIsSubmitting(true)
        setError(None)
        
        // Convert sections dictionary to array of section names
        val sections = includeSections.toList.filter(_._2).map(_._1).toSeq
        
        // Prepare request body
        val requestBody = js.Dynamic.literal(
          title = title,
          description = if (description.trim() == "") null else description,
          analysisId = props.analysisId,
          includeSections = sections.toArray
        )
        
        // Make API request
        val fetchOptions = js.Dynamic.literal(
          method = "POST",
          headers = js.Dynamic.literal(
            "Content-Type" = "application/json"
          ),
          body = js.JSON.stringify(requestBody.asInstanceOf[js.Object])
        )
        
        dom.fetch(
          "/api/v1/reports", 
          fetchOptions.asInstanceOf[RequestInit]
        )
          .`then`[js.Dynamic](response => {
            if (!response.ok.asInstanceOf[Boolean]) {
              throw js.Error(s"Failed to generate report: ${response.statusText.asInstanceOf[String]}")
            }
            response.json().asInstanceOf[js.Promise[js.Dynamic]]
          })
          .`then`[Unit](data => {
            setIsSubmitting(false)
            val reportId = data.id.asInstanceOf[String]
            props.onReportGenerated(reportId)
          })
          .`catch`(error => {
            console.error("Error generating report:", error)
            setIsSubmitting(false)
            setError(Some(s"Failed to generate report: ${error.toString()}"))
          })
      }
      
      // Handle checkbox toggle
      val handleSectionToggle = (section: String) => (event: ReactEventFrom[dom.html.Input]) => {
        val checked = event.target.checked
        setIncludeSections(prev => {
          val updated = js.Dictionary.from(prev.toList)
          updated(section) = checked
          updated
        })
      }
      
      // Render component
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = s"bg-white rounded-lg shadow-lg overflow-hidden ${props.className.getOrElse("")}"
        ).asInstanceOf[js.Object],
        
        // Form header
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-blue-600 px-6 py-4"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "h2",
            js.Dynamic.literal(
              className = "text-white text-xl font-semibold"
            ).asInstanceOf[js.Object],
            "Generate Analysis Report"
          )
        ),
        
        // Form content
        React.createElement(
          "form",
          js.Dynamic.literal(
            onSubmit = handleSubmit,
            className = "p-6 space-y-6"
          ).asInstanceOf[js.Object],
          
          // Title field
          React.createElement(
            "div",
            js.Dynamic.literal().asInstanceOf[js.Object],
            
            React.createElement(
              "label",
              js.Dynamic.literal(
                htmlFor = "report-title",
                className = "block text-sm font-medium text-gray-700 mb-1"
              ).asInstanceOf[js.Object],
              "Report Title *"
            ),
            
            React.createElement(
              "input",
              js.Dynamic.literal(
                id = "report-title",
                `type` = "text",
                value = title,
                onChange = (e: ReactEventFrom[dom.html.Input]) => setTitle(e.target.value),
                className = "w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                required = true,
                disabled = isSubmitting
              ).asInstanceOf[js.Object]
            )
          ),
          
          // Description field
          React.createElement(
            "div",
            js.Dynamic.literal().asInstanceOf[js.Object],
            
            React.createElement(
              "label",
              js.Dynamic.literal(
                htmlFor = "report-description",
                className = "block text-sm font-medium text-gray-700 mb-1"
              ).asInstanceOf[js.Object],
              "Description"
            ),
            
            React.createElement(
              "textarea",
              js.Dynamic.literal(
                id = "report-description",
                value = description,
                onChange = (e: ReactEventFrom[dom.html.TextArea]) => setDescription(e.target.value),
                className = "w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                rows = 3,
                disabled = isSubmitting
              ).asInstanceOf[js.Object]
            )
          ),
          
          // Report sections
          React.createElement(
            "div",
            js.Dynamic.literal().asInstanceOf[js.Object],
            
            React.createElement(
              "h3",
              js.Dynamic.literal(
                className = "text-sm font-medium text-gray-700 mb-2"
              ).asInstanceOf[js.Object],
              "Include Sections:"
            ),
            
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "grid grid-cols-1 sm:grid-cols-2 gap-3"
              ).asInstanceOf[js.Object],
              
              // Section checkboxes
              renderSectionCheckbox("summary", "Summary", includeSections, handleSectionToggle, isSubmitting),
              renderSectionCheckbox("dataDescription", "Data Description", includeSections, handleSectionToggle, isSubmitting),
              renderSectionCheckbox("methodology", "Methodology", includeSections, handleSectionToggle, isSubmitting),
              renderSectionCheckbox("results", "Results", includeSections, handleSectionToggle, isSubmitting),
              renderSectionCheckbox("visualizations", "Visualizations", includeSections, handleSectionToggle, isSubmitting),
              renderSectionCheckbox("interpretations", "Interpretations", includeSections, handleSectionToggle, isSubmitting),
              renderSectionCheckbox("recommendations", "Recommendations", includeSections, handleSectionToggle, isSubmitting)
            )
          ),
          
          // Error message
          error.map(errorText => 
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "text-red-600 text-sm font-medium"
              ).asInstanceOf[js.Object],
              errorText
            )
          ).orNull,
          
          // Form actions
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "flex justify-end space-x-3 mt-6"
            ).asInstanceOf[js.Object],
            
            // Cancel button
            React.createElement(
              "button",
              js.Dynamic.literal(
                `type` = "button",
                onClick = props.onCancel,
                className = "px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500",
                disabled = isSubmitting
              ).asInstanceOf[js.Object],
              "Cancel"
            ),
            
            // Submit button
            React.createElement(
              "button",
              js.Dynamic.literal(
                `type` = "submit",
                className = "px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50",
                disabled = isSubmitting
              ).asInstanceOf[js.Object],
              if (isSubmitting) "Generating..." else "Generate Report"
            )
          )
        )
      )
    }
  }
  
  // Helper function to render a section checkbox
  private def renderSectionCheckbox(
    key: String, 
    label: String, 
    includeSections: js.Dictionary[Boolean],
    handleToggle: String => ReactEventFrom[dom.html.Input] => Unit,
    disabled: Boolean
  ): Element = {
    val id = s"section-$key"
    
    React.createElement(
      "div",
      js.Dynamic.literal(
        className = "flex items-center"
      ).asInstanceOf[js.Object],
      
      React.createElement(
        "input",
        js.Dynamic.literal(
          id = id,
          `type` = "checkbox",
          checked = includeSections.getOrElse(key, false),
          onChange = handleToggle(key),
          disabled = disabled,
          className = "h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
        ).asInstanceOf[js.Object]
      ),
      
      React.createElement(
        "label",
        js.Dynamic.literal(
          htmlFor = id,
          className = "ml-2 text-sm text-gray-700"
        ).asInstanceOf[js.Object],
        label
      )
    )
  }
}

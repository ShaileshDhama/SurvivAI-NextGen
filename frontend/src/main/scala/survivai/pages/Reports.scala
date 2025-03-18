package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*
import org.scalajs.dom
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Success, Failure}
import survivai.components.reports.*
import survivai.services.ReportService
import survivai.components.layout.Layout

object Reports {
  def render(): Element = {
    FC {
      // State hooks
      val (reports, setReports) = useState(js.Array[ReportList.Report]())
      val (isLoading, setIsLoading) = useState(true)
      val (error, setError) = useState[Option[String]](None)
      val (selectedReportId, setSelectedReportId) = useState[Option[String]](None)
      val (showGenerateForm, setShowGenerateForm) = useState(false)
      val (selectedAnalysisId, setSelectedAnalysisId) = useState[Option[String]](None)
      
      // Fetch reports on component mount
      useEffect(() => {
        fetchReports()
        // No cleanup needed
        () => ()
      }, js.Array())
      
      // Fetch reports function
      def fetchReports(): Unit = {
        setIsLoading(true)
        setError(None)
        
        ReportService.getReports().onComplete {
          case Success(fetchedReports) => {
            // Convert reports to the format expected by ReportList component
            val reportListItems = fetchedReports.map(r => 
              ReportList.Report(
                id = r.id,
                title = r.title,
                description = r.description,
                analysisId = r.analysisId,
                createdAt = r.createdAt,
                updatedAt = r.updatedAt
              )
            )
            
            setReports(js.Array(reportListItems: _*))
            setIsLoading(false)
          }
          case Failure(ex) => {
            console.error("Failed to fetch reports:", ex.getMessage)
            setError(Some(s"Failed to fetch reports: ${ex.getMessage}"))
            setIsLoading(false)
          }
        }
      }
      
      // Handle view report
      val handleViewReport = (reportId: String) => {
        setSelectedReportId(Some(reportId))
      }
      
      // Handle delete report
      val handleDeleteReport = (reportId: String) => {
        if (dom.window.confirm("Are you sure you want to delete this report? This action cannot be undone.")) {
          ReportService.deleteReport(reportId).onComplete {
            case Success(_) => {
              // Report deleted successfully, refresh list
              fetchReports()
              // If the deleted report was selected, clear selection
              if (selectedReportId.contains(reportId)) {
                setSelectedReportId(None)
              }
            }
            case Failure(ex) => {
              console.error("Failed to delete report:", ex.getMessage)
              dom.window.alert(s"Failed to delete report: ${ex.getMessage}")
            }
          }
        }
      }
      
      // Handle close report viewer
      val handleCloseReport = () => {
        setSelectedReportId(None)
      }
      
      // Handle show generate form
      val handleShowGenerateForm = (analysisId: String) => {
        setSelectedAnalysisId(Some(analysisId))
        setShowGenerateForm(true)
      }
      
      // Handle cancel report generation
      val handleCancelGeneration = () => {
        setShowGenerateForm(false)
        setSelectedAnalysisId(None)
      }
      
      // Handle report generated
      val handleReportGenerated = (reportId: String) => {
        // Refresh reports list
        fetchReports()
        // Clear form state
        setShowGenerateForm(false)
        setSelectedAnalysisId(None)
        // Show the newly generated report
        setSelectedReportId(Some(reportId))
      }
      
      // Main content based on state
      val content = {
        if (selectedReportId.isDefined) {
          // Show report viewer
          ReportViewer.render(ReportViewer.Props(
            reportId = selectedReportId.get,
            onClose = handleCloseReport,
            className = Some("h-full")
          ))
        } else if (showGenerateForm && selectedAnalysisId.isDefined) {
          // Show report generation form
          ReportGenerationForm.render(ReportGenerationForm.Props(
            analysisId = selectedAnalysisId.get,
            onReportGenerated = handleReportGenerated,
            onCancel = handleCancelGeneration
          ))
        } else {
          // Show reports list
          React.createElement(
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
                "Reports"
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
            ).orNull,
            
            // Reports list
            ReportList.render(ReportList.Props(
              reports = reports,
              isLoading = isLoading,
              onViewReport = handleViewReport,
              onDeleteReport = handleDeleteReport
            ))
          )
        }
      }
      
      // Render with layout
      Layout.render(
        Layout.Props(
          children = content,
          title = "Reports",
          subtitle = "View and manage your survival analysis reports"
        )
      )
    }
  }
}

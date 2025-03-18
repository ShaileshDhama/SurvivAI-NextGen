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
import survivai.components.visualizations.{KaplanMeierCurve, FeatureImportanceViz}

object AnalysisDetail {
  def render(id: String): Element = {
    FC {
      // Use layout context
      val layoutContext = LayoutContext.useLayout()
      
      // States
      val analysisState = React.useState[Option[Analysis.Analysis]](None)
      val analysis = analysisState(0).asInstanceOf[Option[Analysis.Analysis]]
      val setAnalysis = analysisState(1).asInstanceOf[js.Function1[Option[Analysis.Analysis], Unit]]
      
      val datasetState = React.useState[Option[Dataset.Dataset]](None)
      val dataset = datasetState(0).asInstanceOf[Option[Dataset.Dataset]]
      val setDataset = datasetState(1).asInstanceOf[js.Function1[Option[Dataset.Dataset], Unit]]
      
      val loadingState = React.useState[Boolean](true)
      val loading = loadingState(0).asInstanceOf[Boolean]
      val setLoading = loadingState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      val errorState = React.useState[Option[String]](None)
      val error = errorState(0).asInstanceOf[Option[String]]
      val setError = errorState(1).asInstanceOf[js.Function1[Option[String], Unit]]
      
      // Fetch analysis data
      React.useEffect(() => {
        setLoading(true)
        setError(None)
        
        AnalysisService.getAnalysis(id).onComplete {
          case Success(analysis) =>
            setAnalysis(Some(analysis))
            layoutContext.setTitle(analysis.name)
            
            // Fetch associated dataset
            DatasetService.getDataset(analysis.datasetId).onComplete {
              case Success(dataset) =>
                setDataset(Some(dataset))
                setLoading(false)
              case Failure(exception) =>
                setError(Some(s"Error loading dataset: ${exception.getMessage}"))
                setLoading(false)
            }
            
          case Failure(exception) =>
            setError(Some(s"Error loading analysis: ${exception.getMessage}"))
            setLoading(false)
        }
        
        () => ()
      }, js.Array(id))
      
      // Render loading state
      if (loading) {
        return React.createElement(
          "div",
          js.Dynamic.literal(
            className = "flex items-center justify-center py-16"
          ).asInstanceOf[js.Object],
          "Loading analysis..."
        )
      }
      
      // Render error state
      error.foreach { err =>
        return React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-red-50 border border-red-200 text-red-800 rounded-md p-4"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "h3",
            js.Dynamic.literal(
              className = "font-medium"
            ).asInstanceOf[js.Object],
            "Error"
          ),
          
          React.createElement(
            "p",
            null,
            err
          )
        )
      }
      
      // Render not found state
      if (analysis.isEmpty) {
        return React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-yellow-50 border border-yellow-200 text-yellow-800 rounded-md p-4"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "h3",
            js.Dynamic.literal(
              className = "font-medium"
            ).asInstanceOf[js.Object],
            "Analysis Not Found"
          ),
          
          React.createElement(
            "p",
            null,
            s"No analysis found with ID: $id"
          ),
          
          React.createElement(
            "a",
            js.Dynamic.literal(
              href = "#/analyses",
              className = "mt-2 inline-block text-yellow-600 hover:text-yellow-800"
            ).asInstanceOf[js.Object],
            "Back to Analyses"
          )
        )
      }
      
      // Get the analysis and dataset
      val a = analysis.get
      val d = dataset.getOrElse(null)
      
      // Main content
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = "space-y-6"
        ).asInstanceOf[js.Object],
        
        // Header section
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "flex justify-between items-center"
          ).asInstanceOf[js.Object],
          
          // Title and status
          React.createElement(
            "div",
            null,
            
            React.createElement(
              "h1",
              js.Dynamic.literal(
                className = "text-2xl font-bold"
              ).asInstanceOf[js.Object],
              a.name
            ),
            
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = s"inline-block px-2 py-1 mt-2 rounded-full text-xs font-semibold ${getStatusColorClass(a.status)}"
              ).asInstanceOf[js.Object],
              Analysis.Status.toString(a.status)
            )
          ),
          
          // Action buttons
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "flex space-x-3"
            ).asInstanceOf[js.Object],
            
            // Back button
            React.createElement(
              "a",
              js.Dynamic.literal(
                href = "#/analyses",
                className = "px-3 py-1 border border-gray-300 rounded-md text-gray-600 hover:bg-gray-50 transition-colors"
              ).asInstanceOf[js.Object],
              "Back"
            ),
            
            // Export button (if analysis is completed)
            if (a.status == Analysis.Completed) {
              React.createElement(
                "button",
                js.Dynamic.literal(
                  className = "px-3 py-1 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                ).asInstanceOf[js.Object],
                "Export Results"
              )
            } else null
          )
        ),
        
        // Analysis metadata
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "grid grid-cols-1 md:grid-cols-2 gap-6"
          ).asInstanceOf[js.Object],
          
          // Analysis info card
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-white rounded-lg shadow p-6"
            ).asInstanceOf[js.Object],
            
            // Card title
            React.createElement(
              "h2",
              js.Dynamic.literal(
                className = "text-lg font-medium mb-4"
              ).asInstanceOf[js.Object],
              "Analysis Information"
            ),
            
            // Info list
            React.createElement(
              "dl",
              js.Dynamic.literal(
                className = "space-y-2"
              ).asInstanceOf[js.Object],
              
              // Analysis type
              infoItem("Analysis Type", Analysis.AnalysisType.toString(a.analysisType)),
              
              // Dataset name
              infoItem("Dataset", if (d != null) d.name else a.datasetId),
              
              // Created at
              infoItem("Created", formatDate(a.createdAt)),
              
              // Time column
              infoItem("Time Column", a.timeColumn),
              
              // Event column
              infoItem("Event Column", a.eventColumn),
              
              // Description (if provided)
              a.description.map(desc => infoItem("Description", desc)).getOrElse(null)
            )
          ),
          
          // Covariates card
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-white rounded-lg shadow p-6"
            ).asInstanceOf[js.Object],
            
            // Card title
            React.createElement(
              "h2",
              js.Dynamic.literal(
                className = "text-lg font-medium mb-4"
              ).asInstanceOf[js.Object],
              "Covariates"
            ),
            
            // Covariates list
            if (a.covariates.nonEmpty) {
              React.createElement(
                "ul",
                js.Dynamic.literal(
                  className = "list-disc list-inside space-y-1"
                ).asInstanceOf[js.Object],
                
                a.covariates.map { cov =>
                  React.createElement(
                    "li",
                    js.Dynamic.literal(
                      key = cov,
                      className = "text-gray-700"
                    ).asInstanceOf[js.Object],
                    cov
                  )
                }: _*
              )
            } else {
              React.createElement(
                "p",
                js.Dynamic.literal(
                  className = "text-gray-500 italic"
                ).asInstanceOf[js.Object],
                "No covariates"
              )
            }
          )
        ),
        
        // Results section (only if analysis is completed)
        if (a.status == Analysis.Completed) {
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-white rounded-lg shadow p-6"
            ).asInstanceOf[js.Object],
            
            // Section title
            React.createElement(
              "h2",
              js.Dynamic.literal(
                className = "text-lg font-medium mb-6"
              ).asInstanceOf[js.Object],
              "Analysis Results"
            ),
            
            // Results content - depends on analysis type
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "space-y-8"
              ).asInstanceOf[js.Object],
              
              // Kaplan-Meier curve (if available)
              if (a.survivalPoints.nonEmpty) {
                React.createElement(
                  "div",
                  null,
                  
                  React.createElement(
                    "h3",
                    js.Dynamic.literal(
                      className = "text-md font-medium mb-4"
                    ).asInstanceOf[js.Object],
                    "Survival Curve"
                  ),
                  
                  KaplanMeierCurve.render(KaplanMeierCurve.Props(
                    data = a.survivalPoints.toSeq,
                    title = "Kaplan-Meier Survival Curve"
                  ))
                )
              } else null,
              
              // Feature importance (if available)
              if (a.featureImportance.nonEmpty) {
                React.createElement(
                  "div",
                  null,
                  
                  React.createElement(
                    "h3",
                    js.Dynamic.literal(
                      className = "text-md font-medium mb-4"
                    ).asInstanceOf[js.Object],
                    "Feature Importance"
                  ),
                  
                  FeatureImportanceViz.render(FeatureImportanceViz.Props(
                    data = a.featureImportance.toSeq
                  ))
                )
              } else null,
              
              // Metrics (if available)
              if (a.metrics.nonEmpty) {
                React.createElement(
                  "div",
                  null,
                  
                  React.createElement(
                    "h3",
                    js.Dynamic.literal(
                      className = "text-md font-medium mb-4"
                    ).asInstanceOf[js.Object],
                    "Performance Metrics"
                  ),
                  
                  renderMetricsTable(a.metrics.toSeq)
                )
              } else null
            )
          )
        } else if (a.status == Analysis.Running) {
          // Running status
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-white rounded-lg shadow p-6 text-center"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "h3",
              js.Dynamic.literal(
                className = "text-lg font-medium mb-4"
              ).asInstanceOf[js.Object],
              "Analysis Running"
            ),
            
            React.createElement(
              "p",
              js.Dynamic.literal(
                className = "text-gray-600 mb-4"
              ).asInstanceOf[js.Object],
              "Your analysis is currently running. This may take several minutes depending on the dataset size and complexity."
            ),
            
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "w-full bg-gray-200 rounded-full h-2.5 mb-4"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "bg-blue-600 h-2.5 rounded-full animate-pulse",
                  style = js.Dynamic.literal(
                    width = "70%"
                  ).asInstanceOf[js.Object]
                ).asInstanceOf[js.Object]
              )
            )
          )
        } else if (a.status == Analysis.Failed) {
          // Failed status
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-red-50 border border-red-200 text-red-800 rounded-md p-6"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "h3",
              js.Dynamic.literal(
                className = "text-lg font-medium mb-2"
              ).asInstanceOf[js.Object],
              "Analysis Failed"
            ),
            
            React.createElement(
              "p",
              null,
              "There was an error processing this analysis. Please check your inputs and try again."
            ),
            
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "mt-4"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "a",
                js.Dynamic.literal(
                  href = s"#/analyses/new?copy=$id",
                  className = "px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                ).asInstanceOf[js.Object],
                "Try Again with Same Parameters"
              )
            )
          )
        } else {
          // Created but not started
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-white rounded-lg shadow p-6 text-center"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "h3",
              js.Dynamic.literal(
                className = "text-lg font-medium mb-4"
              ).asInstanceOf[js.Object],
              "Analysis Not Started"
            ),
            
            React.createElement(
              "p",
              js.Dynamic.literal(
                className = "text-gray-600 mb-6"
              ).asInstanceOf[js.Object],
              "This analysis has been created but not yet started. Click the button below to begin processing."
            ),
            
            React.createElement(
              "button",
              js.Dynamic.literal(
                className = "px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
              ).asInstanceOf[js.Object],
              "Start Analysis"
            )
          )
        }
      )
    }
  }
  
  // Helper function to format date
  private def formatDate(date: js.Date): String = {
    date.toLocaleDateString("en-US", 
      js.Dynamic.literal(
        year = "numeric",
        month = "short",
        day = "numeric",
        hour = "numeric",
        minute = "numeric"
      ).asInstanceOf[js.Object]
    )
  }
  
  // Helper function to get status color class
  private def getStatusColorClass(status: Analysis.Status): String = status match {
    case Analysis.Created => "bg-gray-100 text-gray-800"
    case Analysis.Running => "bg-blue-100 text-blue-800"
    case Analysis.Completed => "bg-green-100 text-green-800"
    case Analysis.Failed => "bg-red-100 text-red-800"
  }
  
  // Helper function to render info item
  private def infoItem(label: String, value: String): Element = {
    React.createElement(
      "div",
      js.Dynamic.literal(
        className = "py-1"
      ).asInstanceOf[js.Object],
      
      React.createElement(
        "dt",
        js.Dynamic.literal(
          className = "text-sm font-medium text-gray-500"
        ).asInstanceOf[js.Object],
        label
      ),
      
      React.createElement(
        "dd",
        js.Dynamic.literal(
          className = "text-sm text-gray-900"
        ).asInstanceOf[js.Object],
        value
      )
    )
  }
  
  // Helper function to render metrics table
  private def renderMetricsTable(metrics: Seq[(String, Double)]): Element = {
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
          
          React.createElement(
            "th",
            js.Dynamic.literal(
              scope = "col",
              className = "px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
            ).asInstanceOf[js.Object],
            "Metric"
          ),
          
          React.createElement(
            "th",
            js.Dynamic.literal(
              scope = "col",
              className = "px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
            ).asInstanceOf[js.Object],
            "Value"
          )
        )
      ),
      
      // Table body
      React.createElement(
        "tbody",
        js.Dynamic.literal(
          className = "bg-white divide-y divide-gray-200"
        ).asInstanceOf[js.Object],
        
        metrics.map { case (name, value) =>
          React.createElement(
            "tr",
            js.Dynamic.literal(
              key = name
            ).asInstanceOf[js.Object],
            
            // Metric name
            React.createElement(
              "td",
              js.Dynamic.literal(
                className = "px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900"
              ).asInstanceOf[js.Object],
              name
            ),
            
            // Metric value
            React.createElement(
              "td",
              js.Dynamic.literal(
                className = "px-6 py-4 whitespace-nowrap text-sm text-gray-500"
              ).asInstanceOf[js.Object],
              f"$value%.4f"
            )
          )
        }: _*
      )
    )
  }
}

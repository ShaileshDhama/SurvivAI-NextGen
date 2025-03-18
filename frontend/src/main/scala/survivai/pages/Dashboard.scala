package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import survivai.bindings.ReactBindings.*
import survivai.components.*
import survivai.services.{AnalysisService, DatasetService}
import survivai.models.{Analysis, Dataset}
import survivai.contexts.LayoutContext

object Dashboard {
  def render(): Element = {
    FC {
      // Use layout context to set page title
      val layoutContext = LayoutContext.useLayout()
      layoutContext.setTitle("Dashboard")
      
      // State for datasets and analyses
      val datasetsState = React.useState[js.Array[Dataset.Dataset]](js.Array())
      val datasets = datasetsState(0).asInstanceOf[js.Array[Dataset.Dataset]]
      val setDatasets = datasetsState(1).asInstanceOf[js.Function1[js.Array[Dataset.Dataset], Unit]]
      
      val analysesState = React.useState[js.Array[Analysis.Analysis]](js.Array())
      val analyses = analysesState(0).asInstanceOf[js.Array[Analysis.Analysis]]
      val setAnalyses = analysesState(1).asInstanceOf[js.Function1[js.Array[Analysis.Analysis], Unit]]
      
      val loadingState = React.useState[Boolean](true)
      val loading = loadingState(0).asInstanceOf[Boolean]
      val setLoading = loadingState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      // Fetch data on component mount
      React.useEffect(() => {
        // Fetch datasets
        DatasetService.getDatasets(None).foreach { datasets =>
          setDatasets(datasets.toArray[Dataset.Dataset])
        }
        
        // Fetch analyses
        AnalysisService.getAnalyses(None).foreach { analyses =>
          setAnalyses(analyses.toArray[Analysis.Analysis])
          setLoading(false)
        }
        
        () => ()
      }, js.Array())
      
      // Calculate summary stats
      val completedAnalyses = analyses.count(_.status == Analysis.Completed)
      val runningAnalyses = analyses.count(_.status == Analysis.Running)
      
      // Render dashboard
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = "space-y-6"
        ).asInstanceOf[js.Object],
        
        // Stats cards row
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"
          ).asInstanceOf[js.Object],
          
          // Datasets card
          statsCard("Total Datasets", datasets.length.toString, "📊"),
          
          // Analyses card
          statsCard("Total Analyses", analyses.length.toString, "🔍"),
          
          // Completed analyses card
          statsCard("Completed Analyses", completedAnalyses.toString, "✅"),
          
          // Running analyses card
          statsCard("Running Analyses", runningAnalyses.toString, "⏳")
        ),
        
        // Recent activity section
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-white rounded-lg shadow p-4"
          ).asInstanceOf[js.Object],
          
          // Section title
          React.createElement(
            "h2",
            js.Dynamic.literal(
              className = "text-xl font-semibold mb-4"
            ).asInstanceOf[js.Object],
            "Recent Activity"
          ),
          
          // Activity items
          if (!loading && analyses.length > 0) {
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "space-y-3"
              ).asInstanceOf[js.Object],
              
              // Sort analyses by creation date and take most recent 5
              analyses
                .sortBy(_.createdAt.getTime())
                .reverse
                .take(5)
                .map(analysis => activityItem(
                  title = analysis.name, 
                  description = s"${Analysis.Status.toString(analysis.status)} - ${Analysis.AnalysisType.toString(analysis.analysisType)}",
                  timestamp = analysis.createdAt,
                  icon = analysisStatusIcon(analysis.status)
                ))
                .toArray: _*
            )
          } else if (loading) {
            React.createElement(
              "p",
              js.Dynamic.literal(
                className = "text-gray-500 italic"
              ).asInstanceOf[js.Object],
              "Loading..."
            )
          } else {
            React.createElement(
              "p",
              js.Dynamic.literal(
                className = "text-gray-500 italic"
              ).asInstanceOf[js.Object],
              "No recent activity"
            )
          }
        ),
        
        // Quick actions section
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-white rounded-lg shadow p-4"
          ).asInstanceOf[js.Object],
          
          // Section title
          React.createElement(
            "h2",
            js.Dynamic.literal(
              className = "text-xl font-semibold mb-4"
            ).asInstanceOf[js.Object],
            "Quick Actions"
          ),
          
          // Action buttons
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "flex flex-wrap gap-3"
            ).asInstanceOf[js.Object],
            
            actionButton("Upload Dataset", "#/datasets/new", "📤"),
            actionButton("Create Analysis", "#/analyses/new", "🔍"),
            actionButton("View Reports", "#/reports", "📝"),
            actionButton("Create Visualization", "#/visualizations/new", "📊")
          )
        )
      )
    }
  }
  
  // Helper to create stats card
  private def statsCard(title: String, value: String, icon: String): Element = {
    React.createElement(
      "div",
      js.Dynamic.literal(
        className = "bg-white rounded-lg shadow p-4 flex items-center"
      ).asInstanceOf[js.Object],
      
      // Icon
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = "text-3xl mr-4"
        ).asInstanceOf[js.Object],
        icon
      ),
      
      // Content
      React.createElement(
        "div",
        null,
        
        // Title
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "text-sm text-gray-500"
          ).asInstanceOf[js.Object],
          title
        ),
        
        // Value
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "text-2xl font-semibold"
          ).asInstanceOf[js.Object],
          value
        )
      )
    )
  }
  
  // Helper to create activity item
  private def activityItem(title: String, description: String, timestamp: js.Date, icon: String): Element = {
    val formattedDate = new js.Date(timestamp).toLocaleString("en-US", 
      js.Dynamic.literal(
        month = "short",
        day = "numeric",
        hour = "numeric",
        minute = "numeric"
      ).asInstanceOf[js.Object]
    )
    
    React.createElement(
      "div",
      js.Dynamic.literal(
        className = "flex items-start p-3 border-b border-gray-100 last:border-0"
      ).asInstanceOf[js.Object],
      
      // Icon
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = "text-xl mr-3"
        ).asInstanceOf[js.Object],
        icon
      ),
      
      // Content
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = "flex-1"
        ).asInstanceOf[js.Object],
        
        // Title
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "font-medium"
          ).asInstanceOf[js.Object],
          title
        ),
        
        // Description
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "text-sm text-gray-500"
          ).asInstanceOf[js.Object],
          description
        )
      ),
      
      // Timestamp
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = "text-xs text-gray-400"
        ).asInstanceOf[js.Object],
        formattedDate.asInstanceOf[String]
      )
    )
  }
  
  // Helper to create action button
  private def actionButton(text: String, href: String, icon: String): Element = {
    React.createElement(
      "a",
      js.Dynamic.literal(
        href = href,
        className = "bg-blue-50 hover:bg-blue-100 text-blue-700 font-medium py-2 px-4 rounded-md inline-flex items-center transition-colors"
      ).asInstanceOf[js.Object],
      
      // Icon
      React.createElement(
        "span",
        js.Dynamic.literal(
          className = "mr-2"
        ).asInstanceOf[js.Object],
        icon
      ),
      
      // Text
      text
    )
  }
  
  // Helper to get icon for analysis status
  private def analysisStatusIcon(status: Analysis.Status): String = status match {
    case Analysis.Created => "🆕"
    case Analysis.Running => "⏳"
    case Analysis.Completed => "✅"
    case Analysis.Failed => "❌"
  }
}

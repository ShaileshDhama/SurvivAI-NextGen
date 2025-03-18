package survivai.components.visualizations

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*
import survivai.components.*
import survivai.models.Analysis.SurvivalPoint

object KaplanMeierCurve {
  case class Props(
    data: Seq[SurvivalPoint],
    title: String,
    width: Int = 600,
    height: Int = 400,
    showConfidenceInterval: Boolean = true,
    xAxisLabel: String = "Time",
    yAxisLabel: String = "Survival Probability",
    showGrid: Boolean = true,
    showLegend: Boolean = true,
    colorScheme: String = "blues"
  )
  
  def render(props: Props): Element = {
    FC {
      // Create Recharts bindings
      val Recharts = js.Dynamic.global.window.Recharts
      val LineChart = Recharts.LineChart
      val Line = Recharts.Line
      val XAxis = Recharts.XAxis
      val YAxis = Recharts.YAxis
      val CartesianGrid = Recharts.CartesianGrid
      val Tooltip = Recharts.Tooltip
      val Legend = Recharts.Legend
      val ResponsiveContainer = Recharts.ResponsiveContainer
      val Area = Recharts.Area
      
      // Convert data to JS format
      val jsData = props.data.map { point =>
        val obj = js.Dynamic.literal(
          time = point.time,
          survival = point.survival
        )
        
        if (props.showConfidenceInterval) {
          point.upper.foreach(obj.upper = _)
          point.lower.foreach(obj.lower = _)
        }
        
        obj.asInstanceOf[js.Object]
      }.toArray
      
      // Determine colors based on color scheme
      val lineColor = props.colorScheme match {
        case "blues" => "#4299e1"
        case "greens" => "#48bb78"
        case "oranges" => "#ed8936"
        case "purples" => "#9f7aea"
        case "spectral" => "#6366f1"
        case "viridis" => "#0694a2"
        case _ => "#4299e1" // Default to blues
      }
      
      // Create chart container
      val containerProps = js.Dynamic.literal(
        width = "100%",
        height = props.height
      )
      
      // Create chart props
      val chartProps = js.Dynamic.literal(
        data = jsData,
        margin = js.Dynamic.literal(
          top = 20,
          right = 30,
          left = 20,
          bottom = 10
        )
      )
      
      // Chart components
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = "bg-white p-4 rounded-lg shadow"
        ).asInstanceOf[js.Object],
        
        // Title
        React.createElement(
          "h3",
          js.Dynamic.literal(
            className = "text-lg font-semibold mb-4 text-gray-800"
          ).asInstanceOf[js.Object],
          props.title
        ),
        
        // Chart
        React.createElement(
          ResponsiveContainer,
          containerProps.asInstanceOf[js.Object],
          
          React.createElement(
            LineChart,
            chartProps.asInstanceOf[js.Object],
            
            // Grid (conditional)
            if (props.showGrid) {
              React.createElement(
                CartesianGrid,
                js.Dynamic.literal(
                  strokeDasharray = "3 3",
                  stroke = "#f5f5f5"
                ).asInstanceOf[js.Object]
              )
            } else null,
            
            // X Axis
            React.createElement(
              XAxis,
              js.Dynamic.literal(
                dataKey = "time",
                label = js.Dynamic.literal(
                  value = props.xAxisLabel,
                  position = "insideBottomRight",
                  offset = -5
                )
              ).asInstanceOf[js.Object]
            ),
            
            // Y Axis
            React.createElement(
              YAxis,
              js.Dynamic.literal(
                domain = js.Array(0, 1),
                label = js.Dynamic.literal(
                  value = props.yAxisLabel,
                  angle = -90,
                  position = "insideLeft"
                )
              ).asInstanceOf[js.Object]
            ),
            
            // Tooltip
            React.createElement(
              Tooltip,
              js.Dynamic.literal().asInstanceOf[js.Object]
            ),
            
            // Legend (conditional)
            if (props.showLegend) {
              React.createElement(
                Legend,
                js.Dynamic.literal(
                  verticalAlign = "top",
                  align = "right"
                ).asInstanceOf[js.Object]
              )
            } else null,
            
            // Confidence interval if enabled
            if (props.showConfidenceInterval) {
              js.Array(
                // Lower bound
                React.createElement(
                  Line,
                  js.Dynamic.literal(
                    dataKey = "lower",
                    stroke = lineColor,
                    strokeDasharray = "5 5",
                    dot = false,
                    activeDot = false,
                    name = "Lower 95% CI"
                  ).asInstanceOf[js.Object]
                ),
                
                // Upper bound
                React.createElement(
                  Line,
                  js.Dynamic.literal(
                    dataKey = "upper",
                    stroke = lineColor,
                    strokeDasharray = "5 5",
                    dot = false,
                    activeDot = false,
                    name = "Upper 95% CI"
                  ).asInstanceOf[js.Object]
                )
              )
            } else js.Array(),
            
            // Main survival curve
            React.createElement(
              Line,
              js.Dynamic.literal(
                dataKey = "survival",
                stroke = lineColor,
                strokeWidth = 2,
                dot = true,
                name = "Survival Probability"
              ).asInstanceOf[js.Object]
            )
          )
        )
      )
    }
  }
}

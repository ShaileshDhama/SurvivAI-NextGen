package survivai.components.visualizations

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*
import survivai.components.*
import survivai.models.Analysis.FeatureImportance

object FeatureImportanceViz {
  case class Props(
    data: Seq[FeatureImportance],
    title: String = "Feature Importance",
    width: Int = 600,
    height: Int = 400,
    barColor: String = "#8884d8",
    showGrid: Boolean = true,
    colorScheme: String = "blues"
  )
  
  def render(props: Props): Element = {
    FC {
      // Create Recharts bindings
      val Recharts = js.Dynamic.global.window.Recharts
      val BarChart = Recharts.BarChart
      val Bar = Recharts.Bar
      val XAxis = Recharts.XAxis
      val YAxis = Recharts.YAxis
      val CartesianGrid = Recharts.CartesianGrid
      val Tooltip = Recharts.Tooltip
      val ResponsiveContainer = Recharts.ResponsiveContainer
      val Cell = Recharts.Cell
      
      // Sort features by importance and take top 10
      val sortedData = props.data.sortBy(-_.importance).take(10)
      
      // Convert data to JS format
      val jsData = sortedData.map { feat =>
        js.Dynamic.literal(
          feature = feat.feature,
          importance = feat.importance
        ).asInstanceOf[js.Object]
      }.toArray
      
      // Determine color based on the color scheme
      val barColor = props.colorScheme match {
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
        layout = "vertical",
        margin = js.Dynamic.literal(
          top = 20,
          right = 30,
          left = 100,
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
        
        // Description
        React.createElement(
          "p",
          js.Dynamic.literal(
            className = "text-sm text-gray-500 mb-4"
          ).asInstanceOf[js.Object],
          "Higher values indicate features with greater influence on survival outcomes."
        ),
        
        // Chart
        React.createElement(
          ResponsiveContainer,
          containerProps.asInstanceOf[js.Object],
          
          React.createElement(
            BarChart,
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
            
            // X Axis (values)
            React.createElement(
              XAxis,
              js.Dynamic.literal(
                `type` = "number"
              ).asInstanceOf[js.Object]
            ),
            
            // Y Axis (features)
            React.createElement(
              YAxis,
              js.Dynamic.literal(
                dataKey = "feature",
                `type` = "category",
                scale = "band"
              ).asInstanceOf[js.Object]
            ),
            
            // Tooltip
            React.createElement(
              Tooltip,
              js.Dynamic.literal().asInstanceOf[js.Object]
            ),
            
            // Feature bars
            React.createElement(
              Bar,
              js.Dynamic.literal(
                dataKey = "importance",
                fill = barColor,
                label = js.Dynamic.literal(
                  position = "right",
                  formatter = js.Function1 { (value: Double) =>
                    f"$value%.3f"
                  }
                )
              ).asInstanceOf[js.Object]
            )
          )
        )
      )
    }
  }
}

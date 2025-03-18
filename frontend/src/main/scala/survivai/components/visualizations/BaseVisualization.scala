package survivai.components.visualizations

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*
import org.scalajs.dom
import survivai.models.Visualization.{Visualization, VisualizationType}

object BaseVisualization {
  // Props for the component
  case class Props(
    visualization: Visualization,
    width: Option[Int] = None,
    height: Option[Int] = None,
    showControls: Boolean = true,
    onConfigChange: Option[js.Function1[js.Object, Unit]] = None,
    className: Option[String] = None
  )
  
  def render(props: Props): Element = {
    FC {
      // State for configuration
      val (showGrid, setShowGrid) = useState(true)
      val (showLegend, setShowLegend) = useState(true)
      val (colorScheme, setColorScheme) = useState("blueGreen")
      
      // Effect to apply and notify config changes
      useEffect(() => {
        props.onConfigChange.foreach { onChange =>
          // Create config object
          val configObj = js.Dynamic.literal(
            showGrid = showGrid,
            showLegend = showLegend,
            colorScheme = colorScheme
          )
          
          onChange(configObj.asInstanceOf[js.Object])
        }
        
        () => ()
      }, js.Array(showGrid, showLegend, colorScheme))
      
      // Initialize config from visualization
      useEffect(() => {
        if (props.visualization != null && props.visualization.config != null) {
          val config = props.visualization.config
          
          if (!js.isUndefined(config.asInstanceOf[js.Dynamic].showGrid)) {
            setShowGrid(config.asInstanceOf[js.Dynamic].showGrid.asInstanceOf[Boolean])
          }
          
          if (!js.isUndefined(config.asInstanceOf[js.Dynamic].showLegend)) {
            setShowLegend(config.asInstanceOf[js.Dynamic].showLegend.asInstanceOf[Boolean])
          }
          
          if (!js.isUndefined(config.asInstanceOf[js.Dynamic].colorScheme)) {
            setColorScheme(config.asInstanceOf[js.Dynamic].colorScheme.asInstanceOf[String])
          }
        }
        
        () => ()
      }, js.Array(props.visualization))
      
      // Helper to render the appropriate visualization based on type
      def renderVisualization(): Element = {
        val width = props.width.getOrElse(600)
        val height = props.height.getOrElse(400)
        
        props.visualization.visualizationType match {
          case VisualizationType.KaplanMeier =>
            // Extract survival data from visualization
            val data = props.visualization.config.asInstanceOf[js.Dynamic].survivalData
            
            if (js.isUndefined(data)) {
              // Display error if data is missing
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "flex items-center justify-center bg-red-50 text-red-500 p-4 rounded-md"
                ).asInstanceOf[js.Object],
                "Error: Missing survival data for Kaplan-Meier curve"
              )
            } else {
              // Render Kaplan-Meier curve with the data
              KaplanMeierCurve.render(
                KaplanMeierCurve.Props(
                  data = data.asInstanceOf[js.Array[KaplanMeierCurve.SurvivalData]],
                  showGrid = showGrid,
                  showLegend = showLegend,
                  colorScheme = Some(colorScheme),
                  width = Some(width),
                  height = Some(height)
                )
              )
            }
            
          case VisualizationType.FeatureImportance =>
            // Extract feature importance data
            val data = props.visualization.config.asInstanceOf[js.Dynamic].features
            
            if (js.isUndefined(data)) {
              // Display error if data is missing
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "flex items-center justify-center bg-red-50 text-red-500 p-4 rounded-md"
                ).asInstanceOf[js.Object],
                "Error: Missing feature data for Feature Importance visualization"
              )
            } else {
              // Render Feature Importance visualization
              FeatureImportanceViz.render(
                FeatureImportanceViz.Props(
                  features = data.asInstanceOf[js.Array[FeatureImportanceViz.FeatureData]],
                  showGrid = showGrid,
                  colorScheme = Some(colorScheme),
                  width = Some(width),
                  height = Some(height)
                )
              )
            }
            
          case VisualizationType.HazardRatio =>
            // Extract hazard ratio data
            val data = props.visualization.config.asInstanceOf[js.Dynamic].hazardRatios
            
            if (js.isUndefined(data)) {
              // Display error if data is missing
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "flex items-center justify-center bg-red-50 text-red-500 p-4 rounded-md"
                ).asInstanceOf[js.Object],
                "Error: Missing data for Hazard Ratio plot"
              )
            } else {
              // Render Hazard Ratio plot
              HazardRatioPlot.render(
                HazardRatioPlot.Props(
                  data = data.asInstanceOf[js.Array[HazardRatioPlot.HazardRatio]],
                  showGrid = showGrid,
                  showConfidenceIntervals = true,
                  colorScheme = Some(colorScheme),
                  width = Some(width),
                  height = Some(height)
                )
              )
            }
            
          case _ =>
            // Fallback for unknown visualization types
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "flex items-center justify-center bg-gray-50 text-gray-500 p-4 rounded-md"
              ).asInstanceOf[js.Object],
              s"Unsupported visualization type: ${props.visualization.visualizationType}"
            )
        }
      }
      
      // Render the component
      val containerClassName = s"visualization-container ${props.className.getOrElse("")} flex flex-col overflow-hidden"
      
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = containerClassName
        ).asInstanceOf[js.Object],
        
        // Visualization title and description
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "mb-4"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "h3",
            js.Dynamic.literal(
              className = "text-lg font-semibold text-gray-800"
            ).asInstanceOf[js.Object],
            props.visualization.title
          ),
          
          // Optional description
          if (!js.isUndefined(props.visualization.description) && props.visualization.description != null) {
            React.createElement(
              "p",
              js.Dynamic.literal(
                className = "text-sm text-gray-600 mt-1"
              ).asInstanceOf[js.Object],
              props.visualization.description.toString
            )
          } else null
        ),
        
        // Main visualization content
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "flex-1 visualization-content"
          ).asInstanceOf[js.Object],
          
          renderVisualization()
        ),
        
        // Visualization controls
        if (props.showControls) {
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "mt-4 pt-3 border-t border-gray-200 visualization-controls"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "flex flex-wrap items-center justify-between gap-4"
              ).asInstanceOf[js.Object],
              
              // Grid toggle
              React.createElement(
                "label",
                js.Dynamic.literal(
                  className = "flex items-center cursor-pointer"
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "input",
                  js.Dynamic.literal(
                    `type` = "checkbox",
                    checked = showGrid,
                    onChange = (_: ReactEventFrom[dom.html.Input]) => setShowGrid(!showGrid),
                    className = "h-4 w-4 text-blue-600 rounded focus:ring-blue-500 mr-2"
                  ).asInstanceOf[js.Object]
                ),
                
                React.createElement(
                  "span",
                  js.Dynamic.literal(
                    className = "text-sm text-gray-700"
                  ).asInstanceOf[js.Object],
                  "Show Grid"
                )
              ),
              
              // Legend toggle (only for visualizations that support legends)
              if (props.visualization.visualizationType == VisualizationType.KaplanMeier) {
                React.createElement(
                  "label",
                  js.Dynamic.literal(
                    className = "flex items-center cursor-pointer"
                  ).asInstanceOf[js.Object],
                  
                  React.createElement(
                    "input",
                    js.Dynamic.literal(
                      `type` = "checkbox",
                      checked = showLegend,
                      onChange = (_: ReactEventFrom[dom.html.Input]) => setShowLegend(!showLegend),
                      className = "h-4 w-4 text-blue-600 rounded focus:ring-blue-500 mr-2"
                    ).asInstanceOf[js.Object]
                  ),
                  
                  React.createElement(
                    "span",
                    js.Dynamic.literal(
                      className = "text-sm text-gray-700"
                    ).asInstanceOf[js.Object],
                    "Show Legend"
                  )
                )
              } else null,
              
              // Color scheme selector
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "flex items-center"
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "span",
                  js.Dynamic.literal(
                    className = "text-sm text-gray-700 mr-2"
                  ).asInstanceOf[js.Object],
                  "Color Scheme:"
                ),
                
                React.createElement(
                  "select",
                  js.Dynamic.literal(
                    value = colorScheme,
                    onChange = (e: ReactEventFrom[dom.html.Select]) => {
                      setColorScheme(e.target.value)
                    },
                    className = "text-sm border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  ).asInstanceOf[js.Object],
                  
                  // Color scheme options
                  js.Array(
                    React.createElement(
                      "option",
                      js.Dynamic.literal(
                        key = "blueGreen",
                        value = "blueGreen"
                      ).asInstanceOf[js.Object],
                      "Blue-Green"
                    ),
                    
                    React.createElement(
                      "option",
                      js.Dynamic.literal(
                        key = "redYellow",
                        value = "redYellow"
                      ).asInstanceOf[js.Object],
                      "Red-Yellow"
                    ),
                    
                    React.createElement(
                      "option",
                      js.Dynamic.literal(
                        key = "purpleOrange",
                        value = "purpleOrange"
                      ).asInstanceOf[js.Object],
                      "Purple-Orange"
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
}

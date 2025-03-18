package survivai.components.visualizations

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*
import org.scalajs.dom
import scala.util.{Success, Failure}

object HazardRatioPlot {
  // Props for the component
  case class Props(
    data: js.Array[HazardRatio],
    title: Option[String] = None,
    showGrid: Boolean = true,
    showConfidenceIntervals: Boolean = true,
    colorScheme: Option[String] = None,
    width: Option[Int] = None,
    height: Option[Int] = None,
    className: Option[String] = None
  )
  
  // Hazard ratio data structure
  case class HazardRatio(
    feature: String,
    ratio: Double,
    lowerCI: Option[Double],
    upperCI: Option[Double]
  )
  
  def render(props: Props): Element = {
    FC {
      // Constants for chart dimensions and styling
      val width = props.width.getOrElse(600)
      val height = props.height.getOrElse(400)
      val margin = js.Dynamic.literal(
        top = 20,
        right = 30,
        bottom = 80,
        left = 80
      )
      
      val chartWidth = width - margin.left.asInstanceOf[Int] - margin.right.asInstanceOf[Int]
      val chartHeight = height - margin.top.asInstanceOf[Int] - margin.bottom.asInstanceOf[Int]
      
      // Colors for the plot
      val colors = props.colorScheme.getOrElse("blueGreen")
      val mainColor = colors match {
        case "blueGreen" => "#3182CE"
        case "redYellow" => "#E53E3E"
        case "purpleOrange" => "#805AD5"
        case _ => "#3182CE"
      }
      
      // Sort data by hazard ratio
      val sortedData = js.Array(props.data.sortBy(_.ratio).reverse: _*)
      
      // Calculate scales
      val maxRatio = Math.max(
        sortedData.map(d => d.upperCI.getOrElse(d.ratio) * 1.1).maxOption.getOrElse(5.0),
        3.0
      )
      
      // Helper function to format label text
      def formatLabel(feature: String): String = {
        // Truncate long feature names
        if (feature.length > 25) feature.substring(0, 22) + "..." else feature
      }
      
      // Create references for D3 manipulations
      val svgRef = React.useRef[dom.Element](null)
      
      // Effect for creating/updating the chart
      React.useEffect(() => {
        // Only proceed if we have data and the DOM element
        if (sortedData.length == 0 || svgRef.current == null) {
          return () => ()
        }
        
        // Clear any existing chart
        val d3 = js.Dynamic.global.d3
        val svg = d3.select(svgRef.current)
        svg.selectAll("*").remove()
        
        // Set up the main SVG and chart group
        val chart = svg
          .attr("width", width)
          .attr("height", height)
          .append("g")
          .attr("transform", s"translate(${margin.left}, ${margin.top})")
        
        // X scale - log scale for hazard ratios
        val xScale = d3.scaleLog()
          .domain(js.Array(0.1, maxRatio))
          .range(js.Array(0, chartWidth))
        
        // Y scale for features
        val yScale = d3.scaleBand()
          .domain(sortedData.map(_.feature).toArray)
          .range(js.Array(0, chartHeight))
          .padding(0.3)
        
        // X axis (bottom)
        val xAxis = chart.append("g")
          .attr("transform", s"translate(0, $chartHeight)")
          .call(d3.axisBottom(xScale).ticks(5).tickFormat(d => {
            val num = d.asInstanceOf[Double]
            if (num < 1) {
              s"${num.toDouble.formatted("%.2f")}"
            } else {
              s"${num.toDouble.formatted("%.1f")}"
            }
          }))
        
        // X axis label
        chart.append("text")
          .attr("text-anchor", "middle")
          .attr("x", chartWidth / 2)
          .attr("y", chartHeight + margin.bottom.asInstanceOf[Int] - 10)
          .attr("fill", "#4A5568")
          .attr("font-size", "14px")
          .text("Hazard Ratio (log scale)")
        
        // Add reference line at 1.0
        chart.append("line")
          .attr("x1", xScale(1))
          .attr("x2", xScale(1))
          .attr("y1", 0)
          .attr("y2", chartHeight)
          .attr("stroke", "#A0AEC0")
          .attr("stroke-width", 1)
          .attr("stroke-dasharray", "5,5")
        
        // Y axis (left)
        val yAxis = chart.append("g")
          .call(d3.axisLeft(yScale).tickFormat(formatLabel))
        
        // Customize axis styling
        svg.selectAll(".domain").attr("stroke", "#E2E8F0")
        svg.selectAll(".tick line").attr("stroke", "#E2E8F0")
        svg.selectAll(".tick text").attr("fill", "#4A5568").attr("font-size", "12px")
        
        // Add grid lines
        if (props.showGrid) {
          chart.append("g")
            .attr("class", "grid")
            .attr("transform", s"translate(0, $chartHeight)")
            .call(
              d3.axisBottom(xScale)
                .tickSize(-chartHeight)
                .tickFormat("") // No labels for grid
            )
            .selectAll("line")
            .attr("stroke", "#EDF2F7")
        }
        
        // Plot hazard ratios
        sortedData.foreach { d =>
          // Draw confidence interval if available and enabled
          if (props.showConfidenceIntervals && d.lowerCI.isDefined && d.upperCI.isDefined) {
            chart.append("line")
              .attr("x1", xScale(d.lowerCI.get))
              .attr("x2", xScale(d.upperCI.get))
              .attr("y1", yScale(d.feature) + yScale.bandwidth() / 2)
              .attr("y2", yScale(d.feature) + yScale.bandwidth() / 2)
              .attr("stroke", mainColor)
              .attr("stroke-width", 2)
              .attr("opacity", 0.7)
            
            // CI end caps
            js.Array(d.lowerCI.get, d.upperCI.get).foreach { ci =>
              chart.append("line")
                .attr("x1", xScale(ci))
                .attr("x2", xScale(ci))
                .attr("y1", yScale(d.feature) + yScale.bandwidth() / 2 - 5)
                .attr("y2", yScale(d.feature) + yScale.bandwidth() / 2 + 5)
                .attr("stroke", mainColor)
                .attr("stroke-width", 2)
                .attr("opacity", 0.7)
            }
          }
          
          // Draw hazard ratio point
          chart.append("circle")
            .attr("cx", xScale(d.ratio))
            .attr("cy", yScale(d.feature) + yScale.bandwidth() / 2)
            .attr("r", 6)
            .attr("fill", mainColor)
            .attr("stroke", "white")
            .attr("stroke-width", 1)
            .attr("opacity", 0.9)
        }
        
        // Add title if provided
        props.title.foreach { title =>
          svg.append("text")
            .attr("text-anchor", "middle")
            .attr("x", width / 2)
            .attr("y", margin.top.asInstanceOf[Int] / 2)
            .attr("font-size", "16px")
            .attr("font-weight", "bold")
            .attr("fill", "#2D3748")
            .text(title)
        }
        
        // Add tooltips
        sortedData.foreach { d =>
          val tooltip = d3.select("body").append("div")
            .attr("class", "hazard-ratio-tooltip")
            .style("position", "absolute")
            .style("visibility", "hidden")
            .style("background-color", "white")
            .style("border", "1px solid #E2E8F0")
            .style("padding", "8px")
            .style("border-radius", "4px")
            .style("box-shadow", "0 2px 4px rgba(0,0,0,0.1)")
            .style("font-size", "12px")
            .style("pointer-events", "none")
            .style("z-index", "1000")
          
          val tipContent = s"${d.feature}: HR ${d.ratio.formatted("%.2f")}"
          val ciContent = if (d.lowerCI.isDefined && d.upperCI.isDefined) {
            s"95% CI: (${d.lowerCI.get.formatted("%.2f")} - ${d.upperCI.get.formatted("%.2f")})"
          } else ""
          
          chart.append("rect")
            .attr("x", 0)
            .attr("y", yScale(d.feature))
            .attr("width", chartWidth)
            .attr("height", yScale.bandwidth())
            .attr("fill", "transparent")
            .style("cursor", "pointer")
            .on("mouseover", () => {
              tooltip
                .style("visibility", "visible")
                .html(s"$tipContent<br/>$ciContent")
            })
            .on("mousemove", event => {
              tooltip
                .style("top", s"${event.pageY - 10}px")
                .style("left", s"${event.pageX + 10}px")
            })
            .on("mouseout", () => {
              tooltip.style("visibility", "hidden")
            })
        }
        
        // Clean up function
        () => {
          val body = d3.select("body")
          body.selectAll(".hazard-ratio-tooltip").remove()
        }
      }, js.Array(width, height, sortedData.length, props.showGrid, props.showConfidenceIntervals, colors))
      
      // Render the SVG element
      val className = s"hazard-ratio-plot ${props.className.getOrElse("")}"
      
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = className
        ).asInstanceOf[js.Object],
        
        React.createElement(
          "svg",
          js.Dynamic.literal(
            ref = svgRef,
            className = "hazard-ratio-svg"
          ).asInstanceOf[js.Object]
        )
      )
    }
  }
  
  // Helper to convert JS object to HazardRatio case class
  def fromJS(obj: js.Dynamic): HazardRatio = {
    val lowerCI = if (js.isUndefined(obj.lowerCI) || obj.lowerCI == null) None 
                 else Some(obj.lowerCI.asInstanceOf[Double])
                 
    val upperCI = if (js.isUndefined(obj.upperCI) || obj.upperCI == null) None 
                 else Some(obj.upperCI.asInstanceOf[Double])
    
    HazardRatio(
      feature = obj.feature.asInstanceOf[String],
      ratio = obj.ratio.asInstanceOf[Double],
      lowerCI = lowerCI,
      upperCI = upperCI
    )
  }
}

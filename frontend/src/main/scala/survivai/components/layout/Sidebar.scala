package survivai.components.layout

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*
import survivai.components.*

object Sidebar {
  def render(): Element = {
    FC {
      val sidebarProps = js.Dynamic.literal(
        className = "w-64 bg-gray-800 text-white p-4 flex flex-col h-full"
      )
      
      React.createElement(
        "div",
        sidebarProps.asInstanceOf[js.Object],
        // Logo/branding
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "text-xl font-bold mb-8"
          ).asInstanceOf[js.Object],
          "SurvivAI-NextGen"
        ),
        
        // Navigation links
        React.createElement(
          "nav",
          js.Dynamic.literal(
            className = "flex-1"
          ).asInstanceOf[js.Object],
          navLink("Dashboard", "#/"),
          navLink("Datasets", "#/datasets"),
          navLink("Analyses", "#/analyses"),
          navLink("Reports", "#/reports"),
          navLink("Visualizations", "#/visualizations"),
          navLink("Profile", "#/profile"),
          navLink("Settings", "#/settings")
        )
      )
    }
  }
  
  private def navLink(text: String, href: String): Element = {
    React.createElement(
      "a",
      js.Dynamic.literal(
        href = href,
        className = "block py-2 px-4 text-gray-300 hover:bg-gray-700 rounded transition-colors"
      ).asInstanceOf[js.Object],
      text
    )
  }
}

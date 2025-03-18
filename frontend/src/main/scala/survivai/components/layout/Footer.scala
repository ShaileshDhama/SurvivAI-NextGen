package survivai.components.layout

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*
import survivai.components.*

object Footer {
  def render(): Element = {
    FC {
      val footerProps = js.Dynamic.literal(
        className = "bg-white border-t py-4 px-6 text-center text-gray-600 text-sm"
      )
      
      React.createElement(
        "footer",
        footerProps.asInstanceOf[js.Object],
        "© 2025 SurvivAI-NextGen. All rights reserved."
      )
    }
  }
}

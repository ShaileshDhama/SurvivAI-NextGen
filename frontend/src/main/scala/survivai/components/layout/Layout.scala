package survivai.components.layout

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*
import survivai.components.*

object Layout {
  case class Props(content: Element)
  
  def render(props: Props): Element = {
    FC {
      val layoutDiv = js.Dynamic.literal(
        className = "flex h-screen bg-gray-100"
      )
      
      React.createElement(
        "div",
        layoutDiv.asInstanceOf[js.Object],
        Sidebar.render(),
        js.Dynamic.literal(
          className = "flex flex-col flex-1 overflow-hidden"
        ).asInstanceOf[js.Object],
        Navbar.render(),
        js.Dynamic.literal(
          className = "flex-1 overflow-auto p-4"
        ).asInstanceOf[js.Object],
        props.content,
        Footer.render()
      )
    }
  }
}

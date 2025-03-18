package survivai

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import org.scalajs.dom
import survivai.bindings.ReactBindings.*

@main
class SurvivAIApp {
  def main(): Unit = {
    val container = dom.document.getElementById("root")
    val root = ReactDOM.createRoot(container)
    root.render(App.render())
  }
}

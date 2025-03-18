package survivai.components

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*

abstract class ReactComponent {
  type Props
  
  def render(props: Props): Element
  
  val component: Component[Props] = (props: Props) => render(props)
}

object ReactComponent {
  def apply[P](renderFn: P => Element): Component[P] = 
    (props: P) => renderFn(props)
}

// Helper for functional components
object FC {
  def apply(render: => Element): Element = render
  
  def apply[P](props: P)(render: P => Element): Element =
    render(props)
}

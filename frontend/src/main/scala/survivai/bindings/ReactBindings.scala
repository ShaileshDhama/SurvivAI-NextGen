package survivai.bindings

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import org.scalajs.dom

object ReactBindings {
  type Element = js.Object
  type Component[Props] = js.Function1[Props, Element]
  type ChildrenProps = js.Object
  
  @js.native
  @JSImport("react", JSImport.Namespace)
  object React extends js.Object {
    def createElement(component: js.Object, props: js.Object): Element = js.native
    def createElement(component: js.Object, props: js.Object, children: Element*): Element = js.native
    
    def createContext[T](defaultValue: T): Context[T] = js.native
    
    def useState[T](initialState: T): js.Array[js.Any] = js.native
    def useEffect(effect: js.Function0[js.Function0[Unit]], deps: js.Array[js.Any]): Unit = js.native
    def useReducer[S, A](reducer: js.Function2[S, A, S], initialState: S): js.Array[js.Any] = js.native
    def useContext[T](context: Context[T]): T = js.native
    def useRef[T](initialValue: T): RefObject[T] = js.native
    def useMemo[T](factory: js.Function0[T], deps: js.Array[js.Any]): T = js.native
    def useCallback[T](callback: js.Function, deps: js.Array[js.Any]): js.Function = js.native
  }
  
  @js.native
  trait Context[T] extends js.Object {
    val Provider: js.Object = js.native
  }
  
  @js.native
  trait RefObject[T] extends js.Object {
    var current: T = js.native
  }
  
  @js.native
  @JSImport("react-dom/client", JSImport.Namespace)
  object ReactDOM extends js.Object {
    def createRoot(container: dom.Element): Root = js.native
  }
  
  @js.native
  trait Root extends js.Object {
    def render(element: Element): Unit = js.native
  }
}

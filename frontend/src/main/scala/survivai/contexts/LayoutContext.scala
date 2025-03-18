package survivai.contexts

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*

object LayoutContext {
  case class LayoutState(
    sidebarOpen: Boolean = true,
    title: String = "Dashboard"
  )
  
  sealed trait LayoutAction
  case object ToggleSidebar extends LayoutAction
  case class SetTitle(title: String) extends LayoutAction
  
  private val initialState = LayoutState()
  private val layoutContext = React.createContext[js.Dynamic](null)
  
  private def layoutReducer(state: LayoutState, action: LayoutAction): LayoutState = action match {
    case ToggleSidebar => state.copy(sidebarOpen = !state.sidebarOpen)
    case SetTitle(title) => state.copy(title = title)
  }
  
  def Provider(children: Element): Element = {
    FC {
      val jsReducer = js.Function2 { (state: js.Dynamic, action: js.Dynamic) =>
        val scalaState = LayoutState(
          sidebarOpen = state.sidebarOpen.asInstanceOf[Boolean],
          title = state.title.asInstanceOf[String]
        )
        
        val actionType = action.`type`.asInstanceOf[String]
        val newState = actionType match {
          case "TOGGLE_SIDEBAR" => layoutReducer(scalaState, ToggleSidebar)
          case "SET_TITLE" => layoutReducer(scalaState, SetTitle(action.payload.asInstanceOf[String]))
          case _ => scalaState
        }
        
        js.Dynamic.literal(
          sidebarOpen = newState.sidebarOpen,
          title = newState.title
        )
      }
      
      val jsInitialState = js.Dynamic.literal(
        sidebarOpen = initialState.sidebarOpen,
        title = initialState.title
      )
      
      val stateAndDispatch = React.useReducer(jsReducer, jsInitialState)
      val state = stateAndDispatch(0).asInstanceOf[js.Dynamic]
      val dispatch = stateAndDispatch(1).asInstanceOf[js.Function1[js.Dynamic, Unit]]
      
      // Helper functions
      val toggleSidebar = js.Function0 { () =>
        dispatch(js.Dynamic.literal(`type` = "TOGGLE_SIDEBAR"))
      }
      
      val setTitle = js.Function1 { (title: String) =>
        dispatch(js.Dynamic.literal(
          `type` = "SET_TITLE",
          payload = title
        ))
      }
      
      // Create context value
      val value = js.Dynamic.literal(
        state = state,
        toggleSidebar = toggleSidebar,
        setTitle = setTitle
      )
      
      // Render provider with value and children
      val providerProps = js.Dynamic.literal(value = value).asInstanceOf[js.Object]
      React.createElement(layoutContext.Provider, providerProps, children)
    }
  }
  
  // Hook to use layout context
  def useLayout(): js.Dynamic = {
    val context = React.useContext(layoutContext)
    if (context == null) {
      throw new RuntimeException("useLayout must be used within a LayoutProvider")
    }
    context
  }
}

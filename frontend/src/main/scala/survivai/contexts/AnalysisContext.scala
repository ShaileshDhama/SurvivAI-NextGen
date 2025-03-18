package survivai.contexts

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*
import org.scalajs.dom
import survivai.models.Analysis
import survivai.services.AnalysisService
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Success, Failure}

object AnalysisContext {
  // Context value type
  case class ContextValue(
    analyses: js.Array[Analysis.Analysis],
    selectedAnalysis: Option[Analysis.Analysis],
    isLoading: Boolean,
    error: Option[String],
    // Context functions
    selectAnalysis: js.Function1[String, Unit],
    loadAnalyses: js.Function0[Unit],
    createAnalysis: js.Function1[Analysis.CreateAnalysisPayload, Unit],
    updateAnalysis: js.Function1[Analysis.Analysis, Unit],
    deleteAnalysis: js.Function1[String, Unit]
  )
  
  // Initial/default context value
  private val defaultContextValue = ContextValue(
    analyses = js.Array(),
    selectedAnalysis = None,
    isLoading = false,
    error = None,
    selectAnalysis = _ => (),
    loadAnalyses = () => (),
    createAnalysis = _ => (),
    updateAnalysis = _ => (),
    deleteAnalysis = _ => ()
  )
  
  // Create the React context
  private val context = React.createContext[ContextValue](defaultContextValue)
  
  // Provider component
  def Provider(children: Element): Element = {
    FC {
      // State hooks
      val (analyses, setAnalyses) = useState[js.Array[Analysis.Analysis]](js.Array())
      val (selectedAnalysis, setSelectedAnalysis) = useState[Option[Analysis.Analysis]](None)
      val (isLoading, setIsLoading) = useState(false)
      val (error, setError) = useState[Option[String]](None)
      
      // Effect to load analyses on mount
      useEffect(() => {
        loadAnalyses()
        () => () // No cleanup needed
      }, js.Array())
      
      // Function to load all analyses
      def loadAnalyses(): Unit = {
        setIsLoading(true)
        setError(None)
        
        AnalysisService.getAnalyses().onComplete {
          case Success(fetchedAnalyses) => {
            setAnalyses(js.Array(fetchedAnalyses: _*))
            setIsLoading(false)
          }
          case Failure(ex) => {
            console.error("Failed to load analyses:", ex.getMessage)
            setError(Some(s"Failed to load analyses: ${ex.getMessage}"))
            setIsLoading(false)
          }
        }
      }
      
      // Function to select an analysis by ID
      def selectAnalysis(id: String): Unit = {
        val analysis = analyses.find(_.id == id)
        if (analysis.isDefined) {
          setSelectedAnalysis(analysis)
        } else {
          // If not in the cache, fetch it
          setIsLoading(true)
          setError(None)
          
          AnalysisService.getAnalysis(id).onComplete {
            case Success(analysis) => {
              setSelectedAnalysis(Some(analysis))
              setIsLoading(false)
            }
            case Failure(ex) => {
              console.error(s"Failed to load analysis $id:", ex.getMessage)
              setError(Some(s"Failed to load analysis: ${ex.getMessage}"))
              setIsLoading(false)
            }
          }
        }
      }
      
      // Function to create a new analysis
      def createAnalysis(payload: Analysis.CreateAnalysisPayload): Unit = {
        setIsLoading(true)
        setError(None)
        
        AnalysisService.createAnalysis(payload).onComplete {
          case Success(newAnalysis) => {
            // Add the new analysis to the list and select it
            setAnalyses(prev => js.Array((newAnalysis +: prev.toSeq): _*))
            setSelectedAnalysis(Some(newAnalysis))
            setIsLoading(false)
          }
          case Failure(ex) => {
            console.error("Failed to create analysis:", ex.getMessage)
            setError(Some(s"Failed to create analysis: ${ex.getMessage}"))
            setIsLoading(false)
          }
        }
      }
      
      // Function to update an existing analysis
      def updateAnalysis(updatedAnalysis: Analysis.Analysis): Unit = {
        setIsLoading(true)
        setError(None)
        
        AnalysisService.updateAnalysis(updatedAnalysis).onComplete {
          case Success(analysis) => {
            // Update both the list and the selected analysis if it's the current one
            setAnalyses(prev => {
              val index = prev.indexWhere(_.id == analysis.id)
              if (index >= 0) {
                val updated = prev.toSeq
                js.Array((updated.patch(index, Seq(analysis), 1)): _*)
              } else {
                prev
              }
            })
            
            // Update selected analysis if it's the one being updated
            if (selectedAnalysis.exists(_.id == analysis.id)) {
              setSelectedAnalysis(Some(analysis))
            }
            
            setIsLoading(false)
          }
          case Failure(ex) => {
            console.error("Failed to update analysis:", ex.getMessage)
            setError(Some(s"Failed to update analysis: ${ex.getMessage}"))
            setIsLoading(false)
          }
        }
      }
      
      // Function to delete an analysis
      def deleteAnalysis(id: String): Unit = {
        setIsLoading(true)
        setError(None)
        
        AnalysisService.deleteAnalysis(id).onComplete {
          case Success(_) => {
            // Remove from list
            setAnalyses(prev => js.Array(prev.filterNot(_.id == id).toSeq: _*))
            
            // Clear selected analysis if it was the one deleted
            if (selectedAnalysis.exists(_.id == id)) {
              setSelectedAnalysis(None)
            }
            
            setIsLoading(false)
          }
          case Failure(ex) => {
            console.error("Failed to delete analysis:", ex.getMessage)
            setError(Some(s"Failed to delete analysis: ${ex.getMessage}"))
            setIsLoading(false)
          }
        }
      }
      
      // Create the context value
      val contextValue = ContextValue(
        analyses = analyses,
        selectedAnalysis = selectedAnalysis,
        isLoading = isLoading,
        error = error,
        selectAnalysis = selectAnalysis,
        loadAnalyses = () => loadAnalyses(),
        createAnalysis = createAnalysis,
        updateAnalysis = updateAnalysis,
        deleteAnalysis = deleteAnalysis
      )
      
      // Provide the context to children
      React.createElement(
        context.Provider,
        js.Dynamic.literal(
          value = contextValue
        ).asInstanceOf[js.Object],
        children
      )
    }
  }
  
  // Hook to use the context
  def useAnalysis(): ContextValue = {
    val value = React.useContext(context)
    if (value == null) {
      throw new Exception("useAnalysis must be used within an AnalysisContext.Provider")
    }
    value
  }
}

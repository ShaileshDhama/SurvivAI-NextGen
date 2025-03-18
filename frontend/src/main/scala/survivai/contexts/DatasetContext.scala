package survivai.contexts

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import survivai.bindings.ReactBindings.*
import survivai.models.Dataset
import survivai.services.DatasetService

object DatasetContext {
  trait ContextValue extends js.Object {
    val datasets: js.Array[Dataset.Dataset]
    val isLoading: Boolean
    val error: js.UndefOr[String]
    val refreshDatasets: js.Function0[Unit]
    val getDataset: js.Function1[String, js.UndefOr[Dataset.Dataset]]
    val selectedDataset: js.UndefOr[Dataset.Dataset]
    val setSelectedDataset: js.Function1[Dataset.Dataset, Unit]
  }

  val context = React.createContext[js.UndefOr[ContextValue]](js.undefined)

  @JSExportTopLevel("DatasetContext")
  val Provider = {
    FC.withChildren { (children: ChildrenType) =>
      val (datasets, setDatasets) = useState(js.Array[Dataset.Dataset]())
      val (isLoading, setIsLoading) = useState(false)
      val (error, setError) = useState[js.UndefOr[String]](js.undefined)
      val (selectedDataset, setSelectedDataset) = useState[js.UndefOr[Dataset.Dataset]](js.undefined)

      val refreshDatasets = () => {
        setIsLoading(true)
        setError(js.undefined)

        DatasetService.getDatasets().foreach { result =>
          setDatasets(js.Array(result: _*))
          setIsLoading(false)
        } recover { case ex =>
          console.error("Failed to fetch datasets:", ex.getMessage)
          setError(s"Failed to load datasets: ${ex.getMessage}")
          setIsLoading(false)
        }
      }

      val getDataset = (id: String): js.UndefOr[Dataset.Dataset] = {
        datasets.find(_.id == id)
      }

      // Fetch datasets on mount
      useEffect(() => {
        refreshDatasets()
        () => ()
      }, js.Array())

      // Create context value
      val value = js.Dynamic.literal(
        datasets = datasets,
        isLoading = isLoading,
        error = error,
        refreshDatasets = refreshDatasets,
        getDataset = getDataset,
        selectedDataset = selectedDataset,
        setSelectedDataset = (dataset: Dataset.Dataset) => setSelectedDataset(dataset)
      ).asInstanceOf[ContextValue]

      React.createElement(
        context.Provider,
        js.Dynamic.literal(value = value).asInstanceOf[js.Object],
        children
      )
    }
  }

  // Hook for components to use the context
  def useDatasetContext(): ContextValue = {
    val contextValue = React.useContext(context)
    if (contextValue.isEmpty) {
      throw new Exception("useDatasetContext must be used within a DatasetContext.Provider")
    }
    contextValue.get
  }
}

package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Success, Failure}
import survivai.bindings.ReactBindings.*
import survivai.components.*
import survivai.services.DatasetService
import survivai.models.Dataset
import survivai.contexts.LayoutContext

object UploadDataset {
  // Upload states
  sealed trait UploadState
  case object Initial extends UploadState
  case object Uploading extends UploadState
  case object Processing extends UploadState
  case object Complete extends UploadState
  case object Error extends UploadState
  
  def render(): Element = {
    FC {
      // Use layout context to set page title
      val layoutContext = LayoutContext.useLayout()
      layoutContext.setTitle("Upload Dataset")
      
      // States
      val nameState = React.useState[String]("")
      val name = nameState(0).asInstanceOf[String]
      val setName = nameState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val descriptionState = React.useState[String]("")
      val description = descriptionState(0).asInstanceOf[String]
      val setDescription = descriptionState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val fileState = React.useState[Option[js.Dynamic]](None)
      val file = fileState(0).asInstanceOf[Option[js.Dynamic]]
      val setFile = fileState(1).asInstanceOf[js.Function1[Option[js.Dynamic], Unit]]
      
      val uploadStateState = React.useState[UploadState](Initial)
      val uploadState = uploadStateState(0).asInstanceOf[UploadState]
      val setUploadState = uploadStateState(1).asInstanceOf[js.Function1[UploadState, Unit]]
      
      val progressState = React.useState[Int](0)
      val progress = progressState(0).asInstanceOf[Int]
      val setProgress = progressState(1).asInstanceOf[js.Function1[Int, Unit]]
      
      val timeColumnState = React.useState[String]("")
      val timeColumn = timeColumnState(0).asInstanceOf[String]
      val setTimeColumn = timeColumnState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val eventColumnState = React.useState[String]("")
      val eventColumn = eventColumnState(0).asInstanceOf[String]
      val setEventColumn = eventColumnState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val uploadedDatasetState = React.useState[Option[Dataset.Dataset]](None)
      val uploadedDataset = uploadedDatasetState(0).asInstanceOf[Option[Dataset.Dataset]]
      val setUploadedDataset = uploadedDatasetState(1).asInstanceOf[js.Function1[Option[Dataset.Dataset], Unit]]
      
      val errorMessageState = React.useState[String]("")
      val errorMessage = errorMessageState(0).asInstanceOf[String]
      val setErrorMessage = errorMessageState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val columnsState = React.useState[js.Array[Dataset.Column]](js.Array())
      val columns = columnsState(0).asInstanceOf[js.Array[Dataset.Column]]
      val setColumns = columnsState(1).asInstanceOf[js.Function1[js.Array[Dataset.Column], Unit]]
      
      // Handle file change
      val handleFileChange = js.Function1 { (e: js.Dynamic) =>
        val files = e.target.files
        if (files.length > 0) {
          val file = files(0)
          setFile(Some(file))
          
          // Auto-fill name from filename if empty
          if (name.isEmpty) {
            val filename = file.name.asInstanceOf[String]
            val filenameParts = filename.split("\\.").toList
            if (filenameParts.length > 1) {
              setName(filenameParts.dropRight(1).mkString("."))
            } else {
              setName(filename)
            }
          }
        } else {
          setFile(None)
        }
      }
      
      // Handle upload
      val handleUpload = js.Function1 { (e: js.Dynamic) =>
        e.preventDefault()
        
        if (name.isEmpty || file.isEmpty) {
          setErrorMessage("Please provide a name and select a file.")
          return ()
        }
        
        setUploadState(Uploading)
        setProgress(0)
        setErrorMessage("")
        
        val formData = js.Dynamic.global.FormData.new()
        formData.append("name", name)
        if (description.nonEmpty) formData.append("description", description)
        formData.append("file", file.get)
        
        DatasetService.uploadDataset(
          formData,
          (percent: Int) => setProgress(percent)
        ).onComplete {
          case Success(dataset) =>
            setUploadState(Processing)
            setUploadedDataset(Some(dataset))
            
            // Fetch schema for setting time/event columns
            DatasetService.getSchema(dataset.id).foreach { columns =>
              setColumns(columns.toArray[Dataset.Column])
              setUploadState(Complete)
            }
            
          case Failure(exception) =>
            setUploadState(Error)
            setErrorMessage(s"Upload failed: ${exception.getMessage}")
        }
      }
      
      // Handle schema update (setting time and event columns)
      val handleSchemaUpdate = js.Function1 { (e: js.Dynamic) =>
        e.preventDefault()
        
        if (timeColumn.isEmpty || eventColumn.isEmpty) {
          setErrorMessage("Please select both time and event columns.")
          return ()
        }
        
        uploadedDataset.foreach { dataset =>
          setUploadState(Processing)
          setErrorMessage("")
          
          DatasetService.updateSchema(
            dataset.id,
            timeColumn,
            eventColumn
          ).onComplete {
            case Success(_) =>
              // Redirect to dataset list
              js.Dynamic.global.window.location.hash = "#/datasets"
              
            case Failure(exception) =>
              setUploadState(Error)
              setErrorMessage(s"Failed to update schema: ${exception.getMessage}")
          }
        }
      }
      
      // Render
      val initialUploadForm = {
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-white rounded-lg shadow p-6"
          ).asInstanceOf[js.Object],
          
          // Title
          React.createElement(
            "h1",
            js.Dynamic.literal(
              className = "text-2xl font-bold mb-6"
            ).asInstanceOf[js.Object],
            "Upload New Dataset"
          ),
          
          // Error message
          if (errorMessage.nonEmpty) {
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "bg-red-50 border border-red-200 text-red-800 rounded-md p-4 mb-6"
              ).asInstanceOf[js.Object],
              errorMessage
            )
          } else null,
          
          // Upload form
          React.createElement(
            "form",
            js.Dynamic.literal(
              onSubmit = handleUpload,
              className = "space-y-4"
            ).asInstanceOf[js.Object],
            
            // Name field
            React.createElement(
              "div",
              null,
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "name",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "Dataset Name *"
              ),
              
              React.createElement(
                "input",
                js.Dynamic.literal(
                  id = "name",
                  type = "text",
                  value = name,
                  onChange = js.Function1((e: js.Dynamic) => setName(e.target.value.asInstanceOf[String])),
                  className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                  required = true
                ).asInstanceOf[js.Object]
              )
            ),
            
            // Description field
            React.createElement(
              "div",
              null,
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "description",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "Description"
              ),
              
              React.createElement(
                "textarea",
                js.Dynamic.literal(
                  id = "description",
                  value = description,
                  onChange = js.Function1((e: js.Dynamic) => setDescription(e.target.value.asInstanceOf[String])),
                  className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                  rows = 3
                ).asInstanceOf[js.Object]
              )
            ),
            
            // File input
            React.createElement(
              "div",
              null,
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "file",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "File (CSV) *"
              ),
              
              React.createElement(
                "input",
                js.Dynamic.literal(
                  id = "file",
                  type = "file",
                  accept = ".csv",
                  onChange = handleFileChange,
                  className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                  required = true
                ).asInstanceOf[js.Object]
              ),
              
              React.createElement(
                "p",
                js.Dynamic.literal(
                  className = "mt-1 text-sm text-gray-500"
                ).asInstanceOf[js.Object],
                "Upload a CSV file containing your survival data. The file must include at least one time column and one event indicator column."
              )
            ),
            
            // Submit button
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "mt-6"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "button",
                js.Dynamic.literal(
                  type = "submit",
                  disabled = uploadState == Uploading || uploadState == Processing,
                  className = s"w-full px-4 py-2 bg-blue-600 text-white rounded-md ${if (uploadState == Uploading || uploadState == Processing) "opacity-70 cursor-not-allowed" else "hover:bg-blue-700"} transition-colors"
                ).asInstanceOf[js.Object],
                "Upload Dataset"
              )
            )
          )
        )
      }
      
      // Create the completed upload form with schema selection
      val completedUploadForm = {
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-white rounded-lg shadow p-6"
          ).asInstanceOf[js.Object],
          
          // Title
          React.createElement(
            "h1",
            js.Dynamic.literal(
              className = "text-2xl font-bold mb-6"
            ).asInstanceOf[js.Object],
            "Dataset Uploaded Successfully"
          ),
          
          // Progress message
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-green-50 border border-green-200 text-green-800 rounded-md p-4 mb-6"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "p",
              js.Dynamic.literal(
                className = "font-medium"
              ).asInstanceOf[js.Object],
              s"Dataset '${uploadedDataset.map(_.name).getOrElse("")}'  has been uploaded."
            ),
            
            React.createElement(
              "p",
              null,
              "Please specify the time and event columns to complete the setup."
            )
          ),
          
          // Error message
          if (errorMessage.nonEmpty) {
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "bg-red-50 border border-red-200 text-red-800 rounded-md p-4 mb-6"
              ).asInstanceOf[js.Object],
              errorMessage
            )
          } else null,
          
          // Schema form
          React.createElement(
            "form",
            js.Dynamic.literal(
              onSubmit = handleSchemaUpdate,
              className = "space-y-4"
            ).asInstanceOf[js.Object],
            
            // Time column field
            React.createElement(
              "div",
              null,
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "timeColumn",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "Time Column (numerical) *"
              ),
              
              React.createElement(
                "select",
                js.Dynamic.literal(
                  id = "timeColumn",
                  value = timeColumn,
                  onChange = js.Function1((e: js.Dynamic) => setTimeColumn(e.target.value.asInstanceOf[String])),
                  className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                  required = true
                ).asInstanceOf[js.Object],
                
                // Default option
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = ""
                  ).asInstanceOf[js.Object],
                  "Select time column"
                ),
                
                // Numeric column options
                columns
                  .filter(_.isNumeric)
                  .map { column =>
                    React.createElement(
                      "option",
                      js.Dynamic.literal(
                        key = column.name,
                        value = column.name
                      ).asInstanceOf[js.Object],
                      column.name
                    )
                  }: _*
              ),
              
              React.createElement(
                "p",
                js.Dynamic.literal(
                  className = "mt-1 text-sm text-gray-500"
                ).asInstanceOf[js.Object],
                "The time column should contain the time until the event or censoring occurs."
              )
            ),
            
            // Event column field
            React.createElement(
              "div",
              null,
              
              React.createElement(
                "label",
                js.Dynamic.literal(
                  htmlFor = "eventColumn",
                  className = "block text-sm font-medium text-gray-700 mb-1"
                ).asInstanceOf[js.Object],
                "Event Column (0/1 indicator) *"
              ),
              
              React.createElement(
                "select",
                js.Dynamic.literal(
                  id = "eventColumn",
                  value = eventColumn,
                  onChange = js.Function1((e: js.Dynamic) => setEventColumn(e.target.value.asInstanceOf[String])),
                  className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                  required = true
                ).asInstanceOf[js.Object],
                
                // Default option
                React.createElement(
                  "option",
                  js.Dynamic.literal(
                    value = ""
                  ).asInstanceOf[js.Object],
                  "Select event column"
                ),
                
                // Column options
                columns.map { column =>
                  React.createElement(
                    "option",
                    js.Dynamic.literal(
                      key = column.name,
                      value = column.name
                    ).asInstanceOf[js.Object],
                    column.name
                  )
                }: _*
              ),
              
              React.createElement(
                "p",
                js.Dynamic.literal(
                  className = "mt-1 text-sm text-gray-500"
                ).asInstanceOf[js.Object],
                "The event column should contain 1 if the event occurred, and 0 if the observation was censored."
              )
            ),
            
            // Submit button
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "mt-6 flex space-x-3"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "button",
                js.Dynamic.literal(
                  type = "submit",
                  disabled = uploadState == Processing,
                  className = s"flex-1 px-4 py-2 bg-blue-600 text-white rounded-md ${if (uploadState == Processing) "opacity-70 cursor-not-allowed" else "hover:bg-blue-700"} transition-colors"
                ).asInstanceOf[js.Object],
                if (uploadState == Processing) "Saving..." else "Save and Continue"
              ),
              
              React.createElement(
                "a",
                js.Dynamic.literal(
                  href = "#/datasets",
                  className = "px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
                ).asInstanceOf[js.Object],
                "Cancel"
              )
            )
          )
        )
      }
      
      // Upload progress
      val uploadProgress = {
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-white rounded-lg shadow p-6 text-center"
          ).asInstanceOf[js.Object],
          
          // Title
          React.createElement(
            "h1",
            js.Dynamic.literal(
              className = "text-2xl font-bold mb-6"
            ).asInstanceOf[js.Object],
            "Uploading Dataset"
          ),
          
          // Progress indicator
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "w-full bg-gray-200 rounded-full h-2.5 mb-6"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "bg-blue-600 h-2.5 rounded-full",
                style = js.Dynamic.literal(
                  width = s"${progress}%"
                ).asInstanceOf[js.Object]
              ).asInstanceOf[js.Object]
            )
          ),
          
          // Progress text
          React.createElement(
            "p",
            js.Dynamic.literal(
              className = "text-gray-600"
            ).asInstanceOf[js.Object],
            if (uploadState == Uploading) {
              s"Uploading... ${progress}%"
            } else {
              "Processing dataset..."
            }
          )
        )
      }
      
      // Error display
      val errorDisplay = {
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "bg-white rounded-lg shadow p-6"
          ).asInstanceOf[js.Object],
          
          // Title
          React.createElement(
            "h1",
            js.Dynamic.literal(
              className = "text-2xl font-bold mb-6 text-red-600"
            ).asInstanceOf[js.Object],
            "Upload Failed"
          ),
          
          // Error message
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-red-50 border border-red-200 text-red-800 rounded-md p-4 mb-6"
            ).asInstanceOf[js.Object],
            errorMessage
          ),
          
          // Retry button
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "flex space-x-3"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "button",
              js.Dynamic.literal(
                onClick = js.Function0(() => setUploadState(Initial)),
                className = "px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
              ).asInstanceOf[js.Object],
              "Try Again"
            ),
            
            React.createElement(
              "a",
              js.Dynamic.literal(
                href = "#/datasets",
                className = "px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
              ).asInstanceOf[js.Object],
              "Back to Datasets"
            )
          )
        )
      }
      
      // Main container
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = "max-w-3xl mx-auto"
        ).asInstanceOf[js.Object],
        
        // Display appropriate component based on upload state
        uploadState match {
          case Initial => initialUploadForm
          case Uploading | Processing => uploadProgress
          case Complete => completedUploadForm
          case Error => errorDisplay
        }
      )
    }
  }
}

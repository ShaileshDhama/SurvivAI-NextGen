package survivai.pages

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Success, Failure}
import survivai.bindings.ReactBindings.*
import survivai.components.*
import survivai.services.{AnalysisService, DatasetService}
import survivai.models.{Analysis, Dataset}
import survivai.contexts.LayoutContext

object NewAnalysis {
  def render(): Element = {
    FC {
      // Use layout context to set page title
      val layoutContext = LayoutContext.useLayout()
      layoutContext.setTitle("Create New Analysis")
      
      // State for datasets
      val datasetsState = React.useState[js.Array[Dataset.Dataset]](js.Array())
      val datasets = datasetsState(0).asInstanceOf[js.Array[Dataset.Dataset]]
      val setDatasets = datasetsState(1).asInstanceOf[js.Function1[js.Array[Dataset.Dataset], Unit]]
      
      // State for selected dataset and schema
      val selectedDatasetState = React.useState[Option[Dataset.Dataset]](None)
      val selectedDataset = selectedDatasetState(0).asInstanceOf[Option[Dataset.Dataset]]
      val setSelectedDataset = selectedDatasetState(1).asInstanceOf[js.Function1[Option[Dataset.Dataset], Unit]]
      
      val columnsState = React.useState[js.Array[Dataset.Column]](js.Array())
      val columns = columnsState(0).asInstanceOf[js.Array[Dataset.Column]]
      val setColumns = columnsState(1).asInstanceOf[js.Function1[js.Array[Dataset.Column], Unit]]
      
      // Form state
      val nameState = React.useState[String]("")
      val name = nameState(0).asInstanceOf[String]
      val setName = nameState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val descriptionState = React.useState[String]("")
      val description = descriptionState(0).asInstanceOf[String]
      val setDescription = descriptionState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val analysisTypeState = React.useState[String]("CoxPH")
      val analysisType = analysisTypeState(0).asInstanceOf[String]
      val setAnalysisType = analysisTypeState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val timeColumnState = React.useState[String]("")
      val timeColumn = timeColumnState(0).asInstanceOf[String]
      val setTimeColumn = timeColumnState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val eventColumnState = React.useState[String]("")
      val eventColumn = eventColumnState(0).asInstanceOf[String]
      val setEventColumn = eventColumnState(1).asInstanceOf[js.Function1[String, Unit]]
      
      val covariatesState = React.useState[js.Array[String]](js.Array())
      val covariates = covariatesState(0).asInstanceOf[js.Array[String]]
      val setCovariates = covariatesState(1).asInstanceOf[js.Function1[js.Array[String], Unit]]
      
      // Loading and error states
      val loadingState = React.useState[Boolean](true)
      val loading = loadingState(0).asInstanceOf[Boolean]
      val setLoading = loadingState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      val submittingState = React.useState[Boolean](false)
      val submitting = submittingState(0).asInstanceOf[Boolean]
      val setSubmitting = submittingState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      val errorState = React.useState[Option[String]](None)
      val error = errorState(0).asInstanceOf[Option[String]]
      val setError = errorState(1).asInstanceOf[js.Function1[Option[String], Unit]]
      
      val successState = React.useState[Boolean](false)
      val success = successState(0).asInstanceOf[Boolean]
      val setSuccess = successState(1).asInstanceOf[js.Function1[Boolean, Unit]]
      
      // Fetch datasets on component mount
      React.useEffect(() => {
        DatasetService.getDatasets(None).foreach { datasets =>
          setDatasets(datasets.toArray[Dataset.Dataset])
          setLoading(false)
        }
        
        () => ()
      }, js.Array())
      
      // Fetch dataset schema when a dataset is selected
      val handleDatasetChange = js.Function1 { (e: js.Dynamic) =>
        val datasetId = e.target.value.asInstanceOf[String]
        if (datasetId.nonEmpty) {
          val dataset = datasets.find(_.id == datasetId)
          setSelectedDataset(dataset)
          
          // Fetch schema
          DatasetService.getSchema(datasetId).foreach { columns =>
            setColumns(columns.toArray[Dataset.Column])
            
            // Set default time and event columns if available
            dataset.foreach { ds =>
              ds.timeColumn.foreach(setTimeColumn)
              ds.eventColumn.foreach(setEventColumn)
            }
          }
        } else {
          setSelectedDataset(None)
          setColumns(js.Array())
        }
      }
      
      // Handle covariate selection
      val handleCovariateChange = js.Function1 { (e: js.Dynamic) =>
        val checkboxes = e.target.form.covariates.asInstanceOf[js.Dynamic]
        
        // Handle both single and multiple checkboxes
        val selectedCovariates = if (js.isUndefined(checkboxes.length)) {
          // Single checkbox
          if (checkboxes.checked.asInstanceOf[Boolean]) {
            js.Array(checkboxes.value.asInstanceOf[String])
          } else {
            js.Array[String]()
          }
        } else {
          // Multiple checkboxes
          val boxes = checkboxes.asInstanceOf[js.Array[js.Dynamic]]
          boxes
            .filter(_.checked.asInstanceOf[Boolean])
            .map(_.value.asInstanceOf[String])
        }
        
        setCovariates(selectedCovariates)
      }
      
      // Handle form submission
      val handleSubmit = js.Function1 { (e: js.Dynamic) =>
        e.preventDefault()
        
        if (name.isEmpty || selectedDataset.isEmpty || timeColumn.isEmpty || eventColumn.isEmpty || covariates.isEmpty) {
          setError(Some("Please fill in all required fields and select at least one covariate."))
          return ()
        }
        
        setSubmitting(true)
        setError(None)
        
        val descOpt = if (description.isEmpty) None else Some(description)
        val payload = Analysis.CreateAnalysisPayload(
          name = name,
          description = descOpt,
          datasetId = selectedDataset.get.id,
          timeColumn = timeColumn,
          eventColumn = eventColumn,
          analysisType = analysisType,
          covariates = covariates.toSeq,
          parameters = js.Dictionary()
        )
        
        AnalysisService.createAnalysis(payload).onComplete {
          case Success(_) =>
            setSubmitting(false)
            setSuccess(true)
            // Reset form
            setName("")
            setDescription("")
            setSelectedDataset(None)
            setTimeColumn("")
            setEventColumn("")
            setCovariates(js.Array())
            
          case Failure(exception) =>
            setSubmitting(false)
            setError(Some(s"Error creating analysis: ${exception.getMessage}"))
        }
      }
      
      // Render form
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = "max-w-4xl mx-auto"
        ).asInstanceOf[js.Object],
        
        // Page title
        React.createElement(
          "h1",
          js.Dynamic.literal(
            className = "text-2xl font-bold mb-6"
          ).asInstanceOf[js.Object],
          "Create New Survival Analysis"
        ),
        
        // Loading indicator
        if (loading) {
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "text-center py-6"
            ).asInstanceOf[js.Object],
            "Loading datasets..."
          )
        } else if (datasets.isEmpty) {
          // No datasets available
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-yellow-50 border border-yellow-200 text-yellow-800 rounded-md p-4 mb-6"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "p",
              js.Dynamic.literal(
                className = "mb-2 font-medium"
              ).asInstanceOf[js.Object],
              "No datasets available"
            ),
            
            React.createElement(
              "p",
              null,
              "Please upload a dataset before creating an analysis."
            ),
            
            React.createElement(
              "a",
              js.Dynamic.literal(
                href = "#/datasets/new",
                className = "mt-2 inline-block px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
              ).asInstanceOf[js.Object],
              "Upload Dataset"
            )
          )
        } else {
          // Form
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "bg-white rounded-lg shadow p-6"
            ).asInstanceOf[js.Object],
            
            // Success message
            if (success) {
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
                  "Analysis created successfully!"
                ),
                
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "mt-2 flex space-x-3"
                  ).asInstanceOf[js.Object],
                  
                  React.createElement(
                    "a",
                    js.Dynamic.literal(
                      href = "#/analyses",
                      className = "px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
                    ).asInstanceOf[js.Object],
                    "View All Analyses"
                  ),
                  
                  React.createElement(
                    "button",
                    js.Dynamic.literal(
                      onClick = js.Function0(() => setSuccess(false)),
                      className = "px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
                    ).asInstanceOf[js.Object],
                    "Create Another"
                  )
                )
              )
            } else {
              React.createElement("div", null)
            },
            
            // Error message
            error.map { errorMsg =>
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "bg-red-50 border border-red-200 text-red-800 rounded-md p-4 mb-6"
                ).asInstanceOf[js.Object],
                errorMsg
              )
            }.getOrElse(React.createElement("div", null)),
            
            // Form
            React.createElement(
              "form",
              js.Dynamic.literal(
                onSubmit = handleSubmit
              ).asInstanceOf[js.Object],
              
              // Basic information section
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "mb-6"
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "h2",
                  js.Dynamic.literal(
                    className = "text-lg font-semibold mb-4 pb-2 border-b"
                  ).asInstanceOf[js.Object],
                  "Basic Information"
                ),
                
                // Analysis name
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "mb-4"
                  ).asInstanceOf[js.Object],
                  
                  React.createElement(
                    "label",
                    js.Dynamic.literal(
                      htmlFor = "name",
                      className = "block text-sm font-medium text-gray-700 mb-1"
                    ).asInstanceOf[js.Object],
                    "Analysis Name *"
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
                
                // Description
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "mb-4"
                  ).asInstanceOf[js.Object],
                  
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
                
                // Dataset selection
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "mb-4"
                  ).asInstanceOf[js.Object],
                  
                  React.createElement(
                    "label",
                    js.Dynamic.literal(
                      htmlFor = "dataset",
                      className = "block text-sm font-medium text-gray-700 mb-1"
                    ).asInstanceOf[js.Object],
                    "Dataset *"
                  ),
                  
                  React.createElement(
                    "select",
                    js.Dynamic.literal(
                      id = "dataset",
                      value = selectedDataset.map(_.id).getOrElse(""),
                      onChange = handleDatasetChange,
                      className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                      required = true
                    ).asInstanceOf[js.Object],
                    
                    // Default option
                    React.createElement(
                      "option",
                      js.Dynamic.literal(
                        value = ""
                      ).asInstanceOf[js.Object],
                      "Select a dataset"
                    ),
                    
                    // Dataset options
                    datasets.map { dataset =>
                      React.createElement(
                        "option",
                        js.Dynamic.literal(
                          key = dataset.id,
                          value = dataset.id
                        ).asInstanceOf[js.Object],
                        dataset.name
                      )
                    }: _*
                  )
                ),
                
                // Analysis type
                React.createElement(
                  "div",
                  null,
                  
                  React.createElement(
                    "label",
                    js.Dynamic.literal(
                      htmlFor = "analysisType",
                      className = "block text-sm font-medium text-gray-700 mb-1"
                    ).asInstanceOf[js.Object],
                    "Analysis Type *"
                  ),
                  
                  React.createElement(
                    "select",
                    js.Dynamic.literal(
                      id = "analysisType",
                      value = analysisType,
                      onChange = js.Function1((e: js.Dynamic) => setAnalysisType(e.target.value.asInstanceOf[String])),
                      className = "w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500",
                      required = true
                    ).asInstanceOf[js.Object],
                    
                    React.createElement(
                      "option",
                      js.Dynamic.literal(
                        value = "CoxPH"
                      ).asInstanceOf[js.Object],
                      "Cox Proportional Hazards"
                    ),
                    
                    React.createElement(
                      "option",
                      js.Dynamic.literal(
                        value = "KaplanMeier"
                      ).asInstanceOf[js.Object],
                      "Kaplan-Meier Survival Analysis"
                    ),
                    
                    React.createElement(
                      "option",
                      js.Dynamic.literal(
                        value = "RandomSurvivalForest"
                      ).asInstanceOf[js.Object],
                      "Random Survival Forest"
                    ),
                    
                    React.createElement(
                      "option",
                      js.Dynamic.literal(
                        value = "WeibullAFT"
                      ).asInstanceOf[js.Object],
                      "Weibull AFT"
                    )
                  )
                )
              ),
              
              // Column selection section (only shown if dataset selected)
              if (selectedDataset.isDefined && columns.nonEmpty) {
                React.createElement(
                  "div",
                  js.Dynamic.literal(
                    className = "mb-6"
                  ).asInstanceOf[js.Object],
                  
                  React.createElement(
                    "h2",
                    js.Dynamic.literal(
                      className = "text-lg font-semibold mb-4 pb-2 border-b"
                    ).asInstanceOf[js.Object],
                    "Column Selection"
                  ),
                  
                  // Time column
                  React.createElement(
                    "div",
                    js.Dynamic.literal(
                      className = "mb-4"
                    ).asInstanceOf[js.Object],
                    
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
                    )
                  ),
                  
                  // Event column
                  React.createElement(
                    "div",
                    js.Dynamic.literal(
                      className = "mb-6"
                    ).asInstanceOf[js.Object],
                    
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
                    )
                  ),
                  
                  // Covariates
                  React.createElement(
                    "div",
                    null,
                    
                    React.createElement(
                      "label",
                      js.Dynamic.literal(
                        className = "block text-sm font-medium text-gray-700 mb-2"
                      ).asInstanceOf[js.Object],
                      "Covariates (select at least one) *"
                    ),
                    
                    React.createElement(
                      "div",
                      js.Dynamic.literal(
                        className = "grid grid-cols-2 sm:grid-cols-3 gap-2 max-h-60 overflow-y-auto border border-gray-300 rounded-md p-3"
                      ).asInstanceOf[js.Object],
                      
                      // Column checkboxes
                      columns
                        // Exclude time and event columns
                        .filter(col => col.name != timeColumn && col.name != eventColumn)
                        .map { column =>
                          React.createElement(
                            "div",
                            js.Dynamic.literal(
                              key = column.name,
                              className = "flex items-center"
                            ).asInstanceOf[js.Object],
                            
                            React.createElement(
                              "input",
                              js.Dynamic.literal(
                                type = "checkbox",
                                id = s"covariate-${column.name}",
                                name = "covariates",
                                value = column.name,
                                checked = covariates.contains(column.name),
                                onChange = handleCovariateChange,
                                className = "mr-2"
                              ).asInstanceOf[js.Object]
                            ),
                            
                            React.createElement(
                              "label",
                              js.Dynamic.literal(
                                htmlFor = s"covariate-${column.name}",
                                className = "text-sm"
                              ).asInstanceOf[js.Object],
                              column.name
                            )
                          )
                        }: _*
                    )
                  )
                )
              } else null,
              
              // Submit button
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "mt-8 flex justify-end"
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "button",
                  js.Dynamic.literal(
                    type = "submit",
                    disabled = submitting,
                    className = s"px-6 py-2 bg-blue-600 text-white rounded-md ${if (submitting) "opacity-70 cursor-not-allowed" else "hover:bg-blue-700"} transition-colors"
                  ).asInstanceOf[js.Object],
                  if (submitting) "Creating Analysis..." else "Create Analysis"
                )
              )
            )
          )
        }
      )
    }
  }
}

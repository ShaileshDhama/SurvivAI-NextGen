package survivai.models

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.scalajs.js.{Date => JSDate}

object Dataset {
  // Column definition
  case class Column(
    name: String,
    dtype: String,
    isNumeric: Boolean,
    uniqueValues: Int,
    missingValues: Int,
    min: Option[Double] = None,
    max: Option[Double] = None,
    mean: Option[Double] = None,
    median: Option[Double] = None
  )
  
  // Dataset summary statistics
  case class Summary(
    totalRows: Int,
    totalColumns: Int,
    numericColumns: Int,
    categoricalColumns: Int,
    columnsWithMissingValues: Int,
    missingCells: Int,
    missingCellsPercent: Double
  )
  
  // Dataset model
  case class Dataset(
    id: String,
    name: String,
    description: Option[String],
    fileName: String,
    uploadedAt: JSDate,
    timeColumn: Option[String],
    eventColumn: Option[String],
    rows: Int,
    columns: Int,
    fileSize: Int
  )
  
  // Upload state
  sealed trait UploadState
  case object NotStarted extends UploadState
  case class Uploading(progress: Double) extends UploadState
  case class Success(dataset: Dataset) extends UploadState
  case class Failed(error: String) extends UploadState
  
  // Dataset filtering parameters
  case class Filters(
    searchTerm: Option[String] = None,
    sortBy: Option[String] = None,
    sortDirection: Option[String] = None
  )
  
  // JavaScript conversions
  implicit class DatasetOps(dataset: Dataset) {
    def toJS: js.Object = {
      val jsObj = js.Dynamic.literal(
        id = dataset.id,
        name = dataset.name,
        fileName = dataset.fileName,
        uploadedAt = dataset.uploadedAt,
        rows = dataset.rows,
        columns = dataset.columns,
        fileSize = dataset.fileSize
      )
      
      dataset.description.foreach(jsObj.description = _)
      dataset.timeColumn.foreach(jsObj.timeColumn = _)
      dataset.eventColumn.foreach(jsObj.eventColumn = _)
      
      jsObj.asInstanceOf[js.Object]
    }
  }
  
  implicit class ColumnOps(column: Column) {
    def toJS: js.Object = {
      val jsObj = js.Dynamic.literal(
        name = column.name,
        dtype = column.dtype,
        isNumeric = column.isNumeric,
        uniqueValues = column.uniqueValues,
        missingValues = column.missingValues
      )
      
      column.min.foreach(jsObj.min = _)
      column.max.foreach(jsObj.max = _)
      column.mean.foreach(jsObj.mean = _)
      column.median.foreach(jsObj.median = _)
      
      jsObj.asInstanceOf[js.Object]
    }
  }
  
  implicit class SummaryOps(summary: Summary) {
    def toJS: js.Object = {
      js.Dynamic.literal(
        totalRows = summary.totalRows,
        totalColumns = summary.totalColumns,
        numericColumns = summary.numericColumns,
        categoricalColumns = summary.categoricalColumns,
        columnsWithMissingValues = summary.columnsWithMissingValues,
        missingCells = summary.missingCells,
        missingCellsPercent = summary.missingCellsPercent
      ).asInstanceOf[js.Object]
    }
  }
  
  // Convert from JS object to Scala case class
  def fromJS(obj: js.Dynamic): Dataset = {
    val description = if (js.isUndefined(obj.description) || obj.description == null) None
                     else Some(obj.description.asInstanceOf[String])
    
    val timeColumn = if (js.isUndefined(obj.timeColumn) || obj.timeColumn == null) None
                    else Some(obj.timeColumn.asInstanceOf[String])
    
    val eventColumn = if (js.isUndefined(obj.eventColumn) || obj.eventColumn == null) None
                     else Some(obj.eventColumn.asInstanceOf[String])
    
    Dataset(
      id = obj.id.asInstanceOf[String],
      name = obj.name.asInstanceOf[String],
      description = description,
      fileName = obj.fileName.asInstanceOf[String],
      uploadedAt = obj.uploadedAt.asInstanceOf[JSDate],
      timeColumn = timeColumn,
      eventColumn = eventColumn,
      rows = obj.rows.asInstanceOf[Int],
      columns = obj.columns.asInstanceOf[Int],
      fileSize = obj.fileSize.asInstanceOf[Int]
    )
  }
  
  def columnFromJS(obj: js.Dynamic): Column = {
    val min = if (js.isUndefined(obj.min) || obj.min == null) None
             else Some(obj.min.asInstanceOf[Double])
    
    val max = if (js.isUndefined(obj.max) || obj.max == null) None
             else Some(obj.max.asInstanceOf[Double])
    
    val mean = if (js.isUndefined(obj.mean) || obj.mean == null) None
              else Some(obj.mean.asInstanceOf[Double])
    
    val median = if (js.isUndefined(obj.median) || obj.median == null) None
                else Some(obj.median.asInstanceOf[Double])
    
    Column(
      name = obj.name.asInstanceOf[String],
      dtype = obj.dtype.asInstanceOf[String],
      isNumeric = obj.isNumeric.asInstanceOf[Boolean],
      uniqueValues = obj.uniqueValues.asInstanceOf[Int],
      missingValues = obj.missingValues.asInstanceOf[Int],
      min = min,
      max = max,
      mean = mean,
      median = median
    )
  }
  
  def summaryFromJS(obj: js.Dynamic): Summary = {
    Summary(
      totalRows = obj.totalRows.asInstanceOf[Int],
      totalColumns = obj.totalColumns.asInstanceOf[Int],
      numericColumns = obj.numericColumns.asInstanceOf[Int],
      categoricalColumns = obj.categoricalColumns.asInstanceOf[Int],
      columnsWithMissingValues = obj.columnsWithMissingValues.asInstanceOf[Int],
      missingCells = obj.missingCells.asInstanceOf[Int],
      missingCellsPercent = obj.missingCellsPercent.asInstanceOf[Double]
    )
  }
}

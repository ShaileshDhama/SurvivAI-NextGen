package survivai.models

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.scalajs.js.{Dictionary, JSON, Date => JSDate}

object Report {
  // Report section types
  sealed trait ReportSection
  case object Summary extends ReportSection
  case object DataDescription extends ReportSection
  case object Methodology extends ReportSection
  case object Results extends ReportSection
  case object Visualizations extends ReportSection
  case object Interpretations extends ReportSection
  case object Recommendations extends ReportSection
  
  object ReportSection {
    def fromString(str: String): ReportSection = str match {
      case "summary" => Summary
      case "dataDescription" => DataDescription
      case "methodology" => Methodology
      case "results" => Results
      case "visualizations" => Visualizations
      case "interpretations" => Interpretations
      case "recommendations" => Recommendations
      case _ => throw new IllegalArgumentException(s"Unknown report section: $str")
    }
    
    def toString(section: ReportSection): String = section match {
      case Summary => "summary"
      case DataDescription => "dataDescription"
      case Methodology => "methodology"
      case Results => "results"
      case Visualizations => "visualizations"
      case Interpretations => "interpretations"
      case Recommendations => "recommendations"
    }
  }
  
  // Report model
  case class Report(
    id: String,
    title: String,
    description: Option[String],
    content: String,
    analysisId: String,
    includedSections: Seq[ReportSection],
    createdAt: JSDate,
    updatedAt: Option[JSDate] = None
  )
  
  // Report creation payload
  case class CreateReportPayload(
    title: String,
    description: Option[String],
    analysisId: String,
    includeSections: Seq[String]
  )
  
  // JavaScript conversions
  implicit class ReportOps(report: Report) {
    def toJS: js.Object = {
      val jsObj = js.Dynamic.literal(
        id = report.id,
        title = report.title,
        description = report.description.getOrElse(null),
        content = report.content,
        analysisId = report.analysisId,
        includedSections = report.includedSections.map(ReportSection.toString(_)).toArray,
        createdAt = report.createdAt
      )
      
      report.updatedAt.foreach(jsObj.updatedAt = _)
      
      jsObj.asInstanceOf[js.Object]
    }
  }
  
  // Convert from JS object to Scala case class
  def fromJS(obj: js.Dynamic): Report = {
    val description = if (js.isUndefined(obj.description) || obj.description == null) None
                     else Some(obj.description.asInstanceOf[String])
    
    val updatedAt = if (js.isUndefined(obj.updatedAt) || obj.updatedAt == null) None
                   else Some(obj.updatedAt.asInstanceOf[JSDate])
    
    val includedSections = if (js.isUndefined(obj.includedSections) || obj.includedSections == null) {
      Seq.empty[ReportSection]
    } else {
      val sectionsArr = obj.includedSections.asInstanceOf[js.Array[String]]
      sectionsArr.map(ReportSection.fromString).toSeq
    }
    
    Report(
      id = obj.id.asInstanceOf[String],
      title = obj.title.asInstanceOf[String],
      description = description,
      content = obj.content.asInstanceOf[String],
      analysisId = obj.analysisId.asInstanceOf[String],
      includedSections = includedSections,
      createdAt = obj.createdAt.asInstanceOf[JSDate],
      updatedAt = updatedAt
    )
  }
}

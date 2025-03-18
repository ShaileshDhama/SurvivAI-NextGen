package survivai.models

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.scalajs.js.{Dictionary, JSON, Date => JSDate}

object Analysis {
  // Analysis types
  sealed trait AnalysisType
  case object CoxPH extends AnalysisType
  case object KaplanMeier extends AnalysisType
  case object RandomSurvivalForest extends AnalysisType
  case object WeibullAFT extends AnalysisType
  
  object AnalysisType {
    def fromString(str: String): AnalysisType = str match {
      case "CoxPH" => CoxPH
      case "KaplanMeier" => KaplanMeier
      case "RandomSurvivalForest" => RandomSurvivalForest
      case "WeibullAFT" => WeibullAFT
      case _ => throw new IllegalArgumentException(s"Unknown analysis type: $str")
    }
    
    def toString(analysisType: AnalysisType): String = analysisType match {
      case CoxPH => "CoxPH"
      case KaplanMeier => "KaplanMeier"
      case RandomSurvivalForest => "RandomSurvivalForest"
      case WeibullAFT => "WeibullAFT"
    }
  }
  
  // Analysis status
  sealed trait Status
  case object Created extends Status
  case object Running extends Status
  case object Completed extends Status
  case object Failed extends Status
  
  object Status {
    def fromString(str: String): Status = str match {
      case "Created" => Created
      case "Running" => Running
      case "Completed" => Completed
      case "Failed" => Failed
      case _ => throw new IllegalArgumentException(s"Unknown status: $str")
    }
    
    def toString(status: Status): String = status match {
      case Created => "Created"
      case Running => "Running"
      case Completed => "Completed"
      case Failed => "Failed"
    }
  }
  
  // Feature importance
  case class FeatureImportance(feature: String, importance: Double)
  
  // Metrics
  case class Metrics(
    concordanceIndex: Double,
    logLikelihood: Double,
    aicScore: Double,
    bicScore: Double
  )
  
  // Survival curve point
  case class SurvivalPoint(
    time: Double,
    survival: Double,
    upper: Option[Double] = None,
    lower: Option[Double] = None
  )
  
  // Analysis model
  case class Analysis(
    id: String,
    name: String,
    description: Option[String],
    datasetId: String,
    timeColumn: String,
    eventColumn: String,
    analysisType: AnalysisType,
    covariates: Seq[String],
    parameters: Dictionary[js.Any],
    status: Status,
    createdAt: JSDate,
    updatedAt: Option[JSDate] = None,
    modelId: Option[String] = None,
    results: Option[Dictionary[js.Any]] = None,
    featureImportance: Option[Seq[FeatureImportance]] = None,
    survivalPoints: Option[Seq[SurvivalPoint]] = None,
    metrics: Option[Metrics] = None
  )
  
  // Analysis creation payload
  case class CreateAnalysisPayload(
    name: String,
    description: Option[String],
    datasetId: String,
    timeColumn: String,
    eventColumn: String,
    analysisType: String,
    covariates: Seq[String],
    parameters: Dictionary[js.Any]
  )
  
  // Analysis filters
  case class Filters(
    searchTerm: Option[String] = None,
    status: Option[Status] = None,
    datasetId: Option[String] = None,
    analysisType: Option[AnalysisType] = None,
    sortBy: Option[String] = None,
    sortDirection: Option[String] = None
  )
  
  // JavaScript conversions
  implicit class AnalysisOps(analysis: Analysis) {
    def toJS: js.Object = {
      val jsObj = js.Dynamic.literal(
        id = analysis.id,
        name = analysis.name,
        description = analysis.description.getOrElse(null),
        datasetId = analysis.datasetId,
        timeColumn = analysis.timeColumn,
        eventColumn = analysis.eventColumn,
        analysisType = AnalysisType.toString(analysis.analysisType),
        covariates = analysis.covariates.toArray,
        parameters = analysis.parameters,
        status = Status.toString(analysis.status),
        createdAt = analysis.createdAt
      )
      
      analysis.updatedAt.foreach(jsObj.updatedAt = _)
      analysis.modelId.foreach(jsObj.modelId = _)
      analysis.results.foreach(jsObj.results = _)
      
      analysis.featureImportance.foreach { features =>
        jsObj.featureImportance = features.map { f =>
          js.Dynamic.literal(
            feature = f.feature,
            importance = f.importance
          ).asInstanceOf[js.Object]
        }.toArray
      }
      
      analysis.survivalPoints.foreach { points =>
        jsObj.survivalPoints = points.map { p =>
          val pointObj = js.Dynamic.literal(
            time = p.time,
            survival = p.survival
          )
          
          p.upper.foreach(pointObj.upper = _)
          p.lower.foreach(pointObj.lower = _)
          
          pointObj.asInstanceOf[js.Object]
        }.toArray
      }
      
      analysis.metrics.foreach { m =>
        jsObj.metrics = js.Dynamic.literal(
          concordanceIndex = m.concordanceIndex,
          logLikelihood = m.logLikelihood,
          aicScore = m.aicScore,
          bicScore = m.bicScore
        ).asInstanceOf[js.Object]
      }
      
      jsObj.asInstanceOf[js.Object]
    }
  }
  
  // Convert from JS object to Scala case class
  def fromJS(obj: js.Dynamic): Analysis = {
    val featureImportance = if (js.isUndefined(obj.featureImportance) || obj.featureImportance == null) {
      None
    } else {
      val featuresArr = obj.featureImportance.asInstanceOf[js.Array[js.Dynamic]]
      Some(featuresArr.map { f =>
        FeatureImportance(
          feature = f.feature.asInstanceOf[String],
          importance = f.importance.asInstanceOf[Double]
        )
      }.toSeq)
    }
    
    val survivalPoints = if (js.isUndefined(obj.survivalPoints) || obj.survivalPoints == null) {
      None
    } else {
      val pointsArr = obj.survivalPoints.asInstanceOf[js.Array[js.Dynamic]]
      Some(pointsArr.map { p =>
        val upper = if (js.isUndefined(p.upper) || p.upper == null) None 
                   else Some(p.upper.asInstanceOf[Double])
        
        val lower = if (js.isUndefined(p.lower) || p.lower == null) None 
                   else Some(p.lower.asInstanceOf[Double])
        
        SurvivalPoint(
          time = p.time.asInstanceOf[Double],
          survival = p.survival.asInstanceOf[Double],
          upper = upper,
          lower = lower
        )
      }.toSeq)
    }
    
    val metrics = if (js.isUndefined(obj.metrics) || obj.metrics == null) {
      None
    } else {
      val m = obj.metrics.asInstanceOf[js.Dynamic]
      Some(Metrics(
        concordanceIndex = m.concordanceIndex.asInstanceOf[Double],
        logLikelihood = m.logLikelihood.asInstanceOf[Double],
        aicScore = m.aicScore.asInstanceOf[Double],
        bicScore = m.bicScore.asInstanceOf[Double]
      ))
    }
    
    val results = if (js.isUndefined(obj.results) || obj.results == null) None
                 else Some(obj.results.asInstanceOf[Dictionary[js.Any]])
    
    val updatedAt = if (js.isUndefined(obj.updatedAt) || obj.updatedAt == null) None
                   else Some(obj.updatedAt.asInstanceOf[JSDate])
    
    val modelId = if (js.isUndefined(obj.modelId) || obj.modelId == null) None
                 else Some(obj.modelId.asInstanceOf[String])
    
    val description = if (js.isUndefined(obj.description) || obj.description == null) None
                     else Some(obj.description.asInstanceOf[String])
    
    Analysis(
      id = obj.id.asInstanceOf[String],
      name = obj.name.asInstanceOf[String],
      description = description,
      datasetId = obj.datasetId.asInstanceOf[String],
      timeColumn = obj.timeColumn.asInstanceOf[String],
      eventColumn = obj.eventColumn.asInstanceOf[String],
      analysisType = AnalysisType.fromString(obj.analysisType.asInstanceOf[String]),
      covariates = obj.covariates.asInstanceOf[js.Array[String]].toSeq,
      parameters = obj.parameters.asInstanceOf[Dictionary[js.Any]],
      status = Status.fromString(obj.status.asInstanceOf[String]),
      createdAt = obj.createdAt.asInstanceOf[JSDate],
      updatedAt = updatedAt,
      modelId = modelId,
      results = results,
      featureImportance = featureImportance,
      survivalPoints = survivalPoints,
      metrics = metrics
    )
  }
}

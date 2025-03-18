package survivai.models

import scala.scalajs.js
import scala.scalajs.js.annotation.*

object Visualization {
  // Enumeration for visualization types
  sealed trait VisualizationType
  case object KaplanMeier extends VisualizationType
  case object FeatureImportance extends VisualizationType
  case object CumulativeHazard extends VisualizationType
  case object StratifiedSurvival extends VisualizationType
  case object TimeDependent extends VisualizationType
  case object Custom extends VisualizationType
  
  object VisualizationType {
    def fromString(str: String): VisualizationType = str match {
      case "kaplan-meier" => KaplanMeier
      case "feature-importance" => FeatureImportance
      case "cumulative-hazard" => CumulativeHazard
      case "stratified-survival" => StratifiedSurvival
      case "time-dependent" => TimeDependent
      case "custom" => Custom
      case _ => KaplanMeier // Default
    }
    
    def toString(vizType: VisualizationType): String = vizType match {
      case KaplanMeier => "kaplan-meier"
      case FeatureImportance => "feature-importance"
      case CumulativeHazard => "cumulative-hazard"
      case StratifiedSurvival => "stratified-survival"
      case TimeDependent => "time-dependent"
      case Custom => "custom"
    }
  }
  
  // Visualization model
  @js.native
  trait Visualization extends js.Object {
    val id: String = js.native
    val title: String = js.native
    val description: js.UndefOr[String] = js.native
    val analysisId: String = js.native
    val visualizationType: VisualizationType = js.native
    val config: js.Object = js.native
    val createdAt: js.Date = js.native
    val updatedAt: js.UndefOr[js.Date] = js.native
  }
  
  // For creating a new visualization
  case class VisualizationCreate(
    title: String,
    description: Option[String],
    analysisId: String,
    visualizationType: VisualizationType,
    config: js.Object
  )
  
  // For updating an existing visualization
  case class VisualizationUpdate(
    id: String,
    title: js.UndefOr[String] = js.undefined,
    description: js.UndefOr[js.UndefOr[String]] = js.undefined,
    visualizationType: js.UndefOr[VisualizationType] = js.undefined,
    config: js.UndefOr[js.Object] = js.undefined
  )
  
  // Implicits for converting between Scala and JS
  implicit def toJs(visualization: Visualization): js.Dynamic = {
    val result = js.Dynamic.literal(
      id = visualization.id,
      title = visualization.title,
      analysisId = visualization.analysisId,
      visualizationType = VisualizationType.toString(visualization.visualizationType),
      config = visualization.config,
      createdAt = visualization.createdAt
    )
    
    visualization.description.toOption.foreach(result.description = _)
    visualization.updatedAt.toOption.foreach(result.updatedAt = _)
    
    result
  }
  
  implicit def fromJs(obj: js.Dynamic): Visualization = {
    js.Dynamic.global.Object.assign(
      js.Dynamic.literal().asInstanceOf[js.Object],
      obj,
      js.Dynamic.literal(
        visualizationType = VisualizationType.fromString(obj.visualizationType.asInstanceOf[String])
      )
    ).asInstanceOf[Visualization]
  }
}

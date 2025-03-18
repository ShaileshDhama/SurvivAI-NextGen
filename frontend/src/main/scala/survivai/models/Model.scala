package survivai.models

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import java.util.Date

object Model {
  
  // Model types
  val CoxPH = "cox-ph"
  val RandomSurvivalForest = "random-survival-forest"
  val KaplanMeier = "kaplan-meier"
  
  // Type for model metrics
  type ModelMetrics = {
    var cIndex: js.UndefOr[Double] = js.undefined
    var ibs: js.UndefOr[Double] = js.undefined
    var aucTime: js.UndefOr[Double] = js.undefined
    var brierScore: js.UndefOr[Double] = js.undefined
  }
  
  @js.native
  trait ModelMetricsNative extends js.Object {
    val cIndex: js.UndefOr[Double] = js.native
    val ibs: js.UndefOr[Double] = js.native
    val aucTime: js.UndefOr[Double] = js.native
    val brierScore: js.UndefOr[Double] = js.native
  }
  
  /**
   * Model case class representing a survival analysis model
   */
  case class Model(
    id: String,
    name: String,
    modelType: String,
    createdAt: Date,
    updatedAt: Date,
    trainingTime: Double,
    analysisId: String,
    parameters: js.Object,
    metrics: ModelMetricsNative,
    description: Option[String] = None
  )
  
  /**
   * Type for creating a new model
   */
  type ModelCreate = {
    var name: String
    var modelType: ModelType
    var analysisId: String
    var parameters: js.Object
    var description: js.UndefOr[String] = js.undefined
  }
  
  /**
   * Type for updating an existing model
   */
  type ModelUpdate = {
    var id: String
    var name: js.UndefOr[String] = js.undefined
    var description: js.UndefOr[String] = js.undefined
    var parameters: js.UndefOr[js.Object] = js.undefined
  }
  
  /**
   * Object representing the type of model
   */
  object ModelType {
    sealed trait ModelType
    case object Cox extends ModelType
    case object RSF extends ModelType
    case object KM extends ModelType
    
    /**
     * Convert a ModelType to a string representation
     */
    def toString(modelType: ModelType): String = modelType match {
      case Cox => CoxPH
      case RSF => RandomSurvivalForest
      case KM => KaplanMeier
    }
    
    /**
     * Convert a string representation to a ModelType
     */
    def fromString(modelType: String): ModelType = modelType match {
      case CoxPH => Cox
      case RandomSurvivalForest => RSF
      case KaplanMeier => KM
      case _ => throw new IllegalArgumentException(s"Unknown model type: $modelType")
    }
  }
  
  /**
   * Convert a JS object to a Model case class
   */
  def fromJs(obj: js.Dynamic): Model = {
    val description = if (js.typeOf(obj.description) != "undefined") {
      Some(obj.description.asInstanceOf[String])
    } else {
      None
    }
    
    Model(
      id = obj.id.asInstanceOf[String],
      name = obj.name.asInstanceOf[String],
      modelType = obj.modelType.asInstanceOf[String],
      createdAt = new Date(obj.createdAt.asInstanceOf[Double]),
      updatedAt = new Date(obj.updatedAt.asInstanceOf[Double]),
      trainingTime = obj.trainingTime.asInstanceOf[Double],
      analysisId = obj.analysisId.asInstanceOf[String],
      parameters = obj.parameters.asInstanceOf[js.Object],
      metrics = obj.metrics.asInstanceOf[ModelMetricsNative],
      description = description
    )
  }
  
  /**
   * Convert a Model case class to a JS object
   */
  def toJs(model: Model): js.Object = {
    js.Dynamic.literal(
      id = model.id,
      name = model.name,
      modelType = model.modelType,
      createdAt = model.createdAt.getTime(),
      updatedAt = model.updatedAt.getTime(),
      trainingTime = model.trainingTime,
      analysisId = model.analysisId,
      parameters = model.parameters,
      metrics = model.metrics,
      description = model.description.orUndefined
    ).asInstanceOf[js.Object]
  }
}

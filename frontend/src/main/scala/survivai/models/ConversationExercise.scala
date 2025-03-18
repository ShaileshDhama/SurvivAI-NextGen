package survivai.models

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import scala.scalajs.js.JSON

// Define the exercise types
sealed trait ExerciseType
case object FreeConversation extends ExerciseType
case object RolePlay extends ExerciseType
case object GrammarPractice extends ExerciseType
case object VocabularyBuilding extends ExerciseType
case object PronunciationPractice extends ExerciseType

object ExerciseType {
  def fromString(str: String): ExerciseType = str match {
    case "free_conversation" => FreeConversation
    case "role_play" => RolePlay
    case "grammar_practice" => GrammarPractice
    case "vocabulary_building" => VocabularyBuilding
    case "pronunciation_practice" => PronunciationPractice
    case _ => FreeConversation
  }
  
  def toString(exerciseType: ExerciseType): String = exerciseType match {
    case FreeConversation => "free_conversation"
    case RolePlay => "role_play"
    case GrammarPractice => "grammar_practice"
    case VocabularyBuilding => "vocabulary_building"
    case PronunciationPractice => "pronunciation_practice"
  }
}

// Supporting models
case class ExamplePhrase(text: String, translation: Option[String] = None)
case class GrammarPoint(rule: String, examples: List[ExamplePhrase])
case class VocabularyItem(term: String, definition: String, examples: List[ExamplePhrase])
case class PronunciationFocus(sound: String, examples: List[String])

// Main model
case class ConversationExercise(
  id: String,
  title: String,
  description: String,
  exerciseType: ExerciseType,
  difficulty: Int, // 1-5 scale
  duration: Int, // in minutes
  examplePhrases: List[ExamplePhrase] = Nil,
  grammarPoints: List[GrammarPoint] = Nil,
  vocabularyItems: List[VocabularyItem] = Nil,
  pronunciationFocus: Option[PronunciationFocus] = None,
  scenario: Option[String] = None,
  instructions: String
)

object ConversationExercise {
  // JavaScript conversions
  implicit class ConversationExerciseOps(exercise: ConversationExercise) {
    def toJS: js.Object = {
      val jsObj = js.Dynamic.literal(
        id = exercise.id,
        title = exercise.title,
        description = exercise.description,
        exerciseType = ExerciseType.toString(exercise.exerciseType),
        difficulty = exercise.difficulty,
        duration = exercise.duration,
        instructions = exercise.instructions
      )
      
      if (exercise.examplePhrases.nonEmpty) {
        jsObj.examplePhrases = exercise.examplePhrases.map { phrase =>
          js.Dynamic.literal(
            text = phrase.text,
            translation = phrase.translation.getOrElse(null)
          ).asInstanceOf[js.Object]
        }.toArray
      }
      
      if (exercise.grammarPoints.nonEmpty) {
        jsObj.grammarPoints = exercise.grammarPoints.map { point =>
          val pointObj = js.Dynamic.literal(
            rule = point.rule
          )
          
          pointObj.examples = point.examples.map { ex =>
            js.Dynamic.literal(
              text = ex.text,
              translation = ex.translation.getOrElse(null)
            ).asInstanceOf[js.Object]
          }.toArray
          
          pointObj.asInstanceOf[js.Object]
        }.toArray
      }
      
      if (exercise.vocabularyItems.nonEmpty) {
        jsObj.vocabularyItems = exercise.vocabularyItems.map { item =>
          val itemObj = js.Dynamic.literal(
            term = item.term,
            definition = item.definition
          )
          
          itemObj.examples = item.examples.map { ex =>
            js.Dynamic.literal(
              text = ex.text,
              translation = ex.translation.getOrElse(null)
            ).asInstanceOf[js.Object]
          }.toArray
          
          itemObj.asInstanceOf[js.Object]
        }.toArray
      }
      
      exercise.pronunciationFocus.foreach { focus =>
        jsObj.pronunciationFocus = js.Dynamic.literal(
          sound = focus.sound,
          examples = focus.examples.toArray
        ).asInstanceOf[js.Object]
      }
      
      exercise.scenario.foreach { scenario =>
        jsObj.scenario = scenario
      }
      
      jsObj.asInstanceOf[js.Object]
    }
  }
  
  // Convert from JS object to Scala case class
  def fromJS(obj: js.Dynamic): ConversationExercise = {
    val examplePhrases = if (js.isUndefined(obj.examplePhrases)) {
      List.empty[ExamplePhrase]
    } else {
      val arr = obj.examplePhrases.asInstanceOf[js.Array[js.Dynamic]]
      arr.map { ex =>
        val translation = if (js.isUndefined(ex.translation) || ex.translation == null) None 
                         else Some(ex.translation.asInstanceOf[String])
        ExamplePhrase(ex.text.asInstanceOf[String], translation)
      }.toList
    }
    
    val grammarPoints = if (js.isUndefined(obj.grammarPoints)) {
      List.empty[GrammarPoint]
    } else {
      val arr = obj.grammarPoints.asInstanceOf[js.Array[js.Dynamic]]
      arr.map { point =>
        val examples = if (js.isUndefined(point.examples)) List.empty[ExamplePhrase]
                      else {
                        val exArr = point.examples.asInstanceOf[js.Array[js.Dynamic]]
                        exArr.map { ex =>
                          val translation = if (js.isUndefined(ex.translation) || ex.translation == null) None 
                                           else Some(ex.translation.asInstanceOf[String])
                          ExamplePhrase(ex.text.asInstanceOf[String], translation)
                        }.toList
                      }
        GrammarPoint(point.rule.asInstanceOf[String], examples)
      }.toList
    }
    
    val vocabularyItems = if (js.isUndefined(obj.vocabularyItems)) {
      List.empty[VocabularyItem]
    } else {
      val arr = obj.vocabularyItems.asInstanceOf[js.Array[js.Dynamic]]
      arr.map { item =>
        val examples = if (js.isUndefined(item.examples)) List.empty[ExamplePhrase]
                      else {
                        val exArr = item.examples.asInstanceOf[js.Array[js.Dynamic]]
                        exArr.map { ex =>
                          val translation = if (js.isUndefined(ex.translation) || ex.translation == null) None 
                                           else Some(ex.translation.asInstanceOf[String])
                          ExamplePhrase(ex.text.asInstanceOf[String], translation)
                        }.toList
                      }
        VocabularyItem(item.term.asInstanceOf[String], item.definition.asInstanceOf[String], examples)
      }.toList
    }
    
    val pronunciationFocus = if (js.isUndefined(obj.pronunciationFocus) || obj.pronunciationFocus == null) {
      None
    } else {
      val focus = obj.pronunciationFocus.asInstanceOf[js.Dynamic]
      val examples = if (js.isUndefined(focus.examples)) List.empty[String]
                    else focus.examples.asInstanceOf[js.Array[String]].toList
      Some(PronunciationFocus(focus.sound.asInstanceOf[String], examples))
    }
    
    val scenario = if (js.isUndefined(obj.scenario) || obj.scenario == null) None
                  else Some(obj.scenario.asInstanceOf[String])
    
    ConversationExercise(
      id = obj.id.asInstanceOf[String],
      title = obj.title.asInstanceOf[String],
      description = obj.description.asInstanceOf[String],
      exerciseType = ExerciseType.fromString(obj.exerciseType.asInstanceOf[String]),
      difficulty = obj.difficulty.asInstanceOf[Int],
      duration = obj.duration.asInstanceOf[Int],
      examplePhrases = examplePhrases,
      grammarPoints = grammarPoints,
      vocabularyItems = vocabularyItems,
      pronunciationFocus = pronunciationFocus,
      scenario = scenario,
      instructions = obj.instructions.asInstanceOf[String]
    )
  }
  
  // Serialization to JSON string
  def toJsonString(exercise: ConversationExercise): String = {
    JSON.stringify(exercise.toJS)
  }
  
  // Deserialization from JSON string
  def fromJsonString(jsonStr: String): ConversationExercise = {
    val obj = JSON.parse(jsonStr).asInstanceOf[js.Dynamic]
    fromJS(obj)
  }
}

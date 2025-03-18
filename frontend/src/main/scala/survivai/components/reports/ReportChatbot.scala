package survivai.components.reports

import scala.scalajs.js
import scala.scalajs.js.annotation.*
import survivai.bindings.ReactBindings.*
import org.scalajs.dom
import java.util.UUID

object ReportChatbot {
  // Message type definition
  sealed trait MessageType
  case object User extends MessageType
  case object Bot extends MessageType
  case object System extends MessageType
  
  // Message record
  case class Message(
    id: String,
    text: String,
    messageType: MessageType,
    timestamp: String
  )
  
  // Component props
  case class Props(
    reportId: Option[String] = None,
    modelId: String,
    onClose: Option[js.Function0[Unit]] = None,
    className: Option[String] = None
  )
  
  def render(props: Props): Element = {
    FC {
      // State hooks
      val (messages, setMessages) = useState(js.Array(
        Message(
          id = "welcome",
          text = "Hello! I'm your AI assistant. I can help you understand the survival analysis report and answer questions about risk factors. What would you like to know?",
          messageType = Bot,
          timestamp = new js.Date().toISOString()
        )
      ))
      
      val (userInput, setUserInput) = useState("")
      val (isLoading, setIsLoading) = useState(false)
      val messagesEndRef = useRef[dom.html.Element](null)
      
      // Scroll to bottom of chat when new messages arrive
      useEffect(() => {
        if (messagesEndRef.current != null) {
          messagesEndRef.current.scrollIntoView(js.Dynamic.literal(behavior = "smooth").asInstanceOf[js.Object])
        }
      }, js.Array(messages))
      
      // Handle user input submission
      val handleSubmit = (event: ReactEventFrom[dom.html.Form]) => {
        event.preventDefault()
        
        if (userInput.trim() != "") {
          // Add user message to chat
          val timestamp = new js.Date().getTime()
          val newUserMessage = Message(
            id = s"user_${timestamp}",
            text = userInput,
            messageType = User,
            timestamp = new js.Date().toISOString()
          )
          
          setMessages(prevMessages => js.Array(prevMessages.toSeq ++ Seq(newUserMessage): _*))
          setUserInput("")
          setIsLoading(true)
          
          // Prepare request to chatbot API
          val requestBody = js.Dynamic.literal(
            question = userInput,
            model_id = props.modelId
          )
          
          props.reportId.foreach(id => requestBody.report_id = id)
          
          // Call chatbot API
          val fetchOptions = js.Dynamic.literal(
            method = "POST",
            headers = js.Dynamic.literal(
              "Content-Type" = "application/json"
            ),
            body = js.JSON.stringify(requestBody.asInstanceOf[js.Object])
          )
          
          dom.fetch(
            "/api/v1/reports/chatbot/question", 
            fetchOptions.asInstanceOf[RequestInit]
          )
            .`then`[js.Dynamic](response => response.json().asInstanceOf[js.Promise[js.Dynamic]])
            .`then`[Unit](data => {
              val botResponse = data.response.asInstanceOf[String]
              
              // Add bot response to chat
              val newBotMessage = Message(
                id = s"bot_${new js.Date().getTime()}",
                text = botResponse,
                messageType = Bot,
                timestamp = new js.Date().toISOString()
              )
              
              setMessages(prevMessages => js.Array(prevMessages.toSeq ++ Seq(newBotMessage): _*))
              setIsLoading(false)
            })
            .`catch`(error => {
              console.error("Error calling chatbot API:", error)
              
              // Add error message to chat
              val errorMessage = Message(
                id = s"error_${new js.Date().getTime()}",
                text = "Sorry, I encountered an error. Please try again later.",
                messageType = System,
                timestamp = new js.Date().toISOString()
              )
              
              setMessages(prevMessages => js.Array(prevMessages.toSeq ++ Seq(errorMessage): _*))
              setIsLoading(false)
            })
        }
      }
      
      // Format message timestamp
      val formatTimestamp = (timestamp: String): String => {
        try {
          val date = new js.Date(timestamp)
          val hours = date.getHours()
          val minutes = date.getMinutes()
          val formattedHours = if (hours < 10) s"0$hours" else hours.toString
          val formattedMinutes = if (minutes < 10) s"0$minutes" else minutes.toString
          s"$formattedHours:$formattedMinutes"
        } catch {
          case _: Throwable => "--:--"
        }
      }
      
      // Container class
      val containerClass = s"flex flex-col h-full rounded-lg shadow-lg bg-white ${props.className.getOrElse("")}"
      
      // Render component
      React.createElement(
        "div",
        js.Dynamic.literal(
          className = containerClass
        ).asInstanceOf[js.Object],
        
        // Chatbot header
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "border-b border-gray-200 p-4 flex justify-between items-center bg-gray-50"
          ).asInstanceOf[js.Object],
          
          // Header content - title and icon
          React.createElement(
            "div",
            js.Dynamic.literal(
              className = "flex items-center"
            ).asInstanceOf[js.Object],
            
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center mr-2"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "span",
                js.Dynamic.literal(
                  className = "text-white text-sm font-bold"
                ).asInstanceOf[js.Object],
                "AI"
              )
            ),
            
            React.createElement(
              "span",
              js.Dynamic.literal(
                className = "font-medium"
              ).asInstanceOf[js.Object],
              "Report Assistant"
            )
          ),
          
          // Close button (if onClose prop provided)
          props.onClose.map { closeHandler =>
            React.createElement(
              "button",
              js.Dynamic.literal(
                className = "text-gray-500 hover:text-gray-700",
                onClick = closeHandler
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "span",
                js.Dynamic.literal(
                  className = "sr-only"
                ).asInstanceOf[js.Object],
                "Close"
              ),
              
              // X icon
              React.createElement(
                "svg",
                js.Dynamic.literal(
                  className = "h-5 w-5",
                  fill = "none",
                  viewBox = "0 0 24 24",
                  stroke = "currentColor"
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "path",
                  js.Dynamic.literal(
                    strokeLinecap = "round",
                    strokeLinejoin = "round",
                    strokeWidth = 2,
                    d = "M6 18L18 6M6 6l12 12"
                  ).asInstanceOf[js.Object]
                )
              )
            )
          }.orNull
        ),
        
        // Message container
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "flex-1 p-4 overflow-y-auto flex flex-col space-y-4"
          ).asInstanceOf[js.Object],
          
          // Render messages
          messages.map { (message, index) =>
            val isUser = message.messageType == User
            val messageClass = message.messageType match {
              case User => "bg-blue-100 text-blue-800 ml-auto"
              case Bot => "bg-gray-100 text-gray-800"
              case System => "bg-red-100 text-red-800 mx-auto"
            }
            
            val containerClass = message.messageType match {
              case User => "flex justify-end"
              case _ => "flex justify-start"
            }
            
            React.createElement(
              "div",
              js.Dynamic.literal(
                key = message.id,
                className = containerClass
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = s"rounded-lg px-4 py-2 max-w-xs sm:max-w-md break-words ${messageClass}"
                ).asInstanceOf[js.Object],
                
                // Message text
                React.createElement(
                  "p",
                  js.Dynamic.literal().asInstanceOf[js.Object],
                  message.text
                ),
                
                // Message timestamp
                React.createElement(
                  "span",
                  js.Dynamic.literal(
                    className = "text-xs text-gray-500 mt-1 block text-right"
                  ).asInstanceOf[js.Object],
                  formatTimestamp(message.timestamp)
                )
              )
            )
          },
          
          // Loading indicator
          if (isLoading) {
            React.createElement(
              "div",
              js.Dynamic.literal(
                className = "flex justify-start"
              ).asInstanceOf[js.Object],
              
              React.createElement(
                "div",
                js.Dynamic.literal(
                  className = "bg-gray-100 rounded-lg px-4 py-2 flex items-center space-x-1"
                ).asInstanceOf[js.Object],
                
                (1 to 3).map { i =>
                  React.createElement(
                    "div",
                    js.Dynamic.literal(
                      key = s"dot-$i",
                      className = "bg-gray-500 rounded-full w-2 h-2 animate-bounce",
                      style = js.Dynamic.literal(
                        animationDelay = s"${i * 0.1}s"
                      )
                    ).asInstanceOf[js.Object]
                  )
                }
              )
            )
          } else null,
          
          // Reference for auto-scrolling
          React.createElement(
            "div",
            js.Dynamic.literal(
              ref = messagesEndRef
            ).asInstanceOf[js.Object]
          )
        ),
        
        // Input form
        React.createElement(
          "div",
          js.Dynamic.literal(
            className = "border-t border-gray-200 p-3"
          ).asInstanceOf[js.Object],
          
          React.createElement(
            "form",
            js.Dynamic.literal(
              onSubmit = handleSubmit,
              className = "flex space-x-2"
            ).asInstanceOf[js.Object],
            
            // Text input
            React.createElement(
              "input",
              js.Dynamic.literal(
                `type` = "text",
                className = "flex-1 border border-gray-300 rounded-full py-2 px-4 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500",
                placeholder = "Type your question here...",
                value = userInput,
                onChange = (e: ReactEventFrom[dom.html.Input]) => setUserInput(e.target.value),
                disabled = isLoading
              ).asInstanceOf[js.Object]
            ),
            
            // Send button
            React.createElement(
              "button",
              js.Dynamic.literal(
                `type` = "submit",
                className = "bg-blue-600 text-white rounded-full p-2 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50",
                disabled = isLoading || userInput.trim() == ""
              ).asInstanceOf[js.Object],
              
              // Send icon
              React.createElement(
                "svg",
                js.Dynamic.literal(
                  className = "h-5 w-5",
                  fill = "none",
                  viewBox = "0 0 24 24",
                  stroke = "currentColor"
                ).asInstanceOf[js.Object],
                
                React.createElement(
                  "path",
                  js.Dynamic.literal(
                    strokeLinecap = "round",
                    strokeLinejoin = "round",
                    strokeWidth = 2,
                    d = "M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                  ).asInstanceOf[js.Object]
                )
              )
            )
          )
        )
      )
    }
  }
}

# Libraries ----
library(shiny)
library(shinydashboard)



# UI ----
ui <- fluidPage(
  title = "Welcome!",
  titlePanel("Title Here"),
  "Hello World",
  br(),
  
  selectInput(
    inputId = "first_input",
    label = "Select a Number",
    choices = c(1,2,3,4,5,6,7,8,9,10)
    #multiple = TRUE,
    #selected = 7
  ),
  
  numericInput(
    inputId = "second_input",
    label = "Select a Number",
    value = 5,
    min = 1, max = 20, step = 1
  ),
  
  verbatimTextOutput("power")
)

server <- function(input, output) {
  
  observe({
    print( as.numeric( input$first_input ) ** input$second_input)
  })
  
  make_power <- function(x, y) { as.numeric(x) ** as.numeric(y) }
  output$power <- renderText({ make_power( input$first_input, input$second_input )})
  
  #output$power <- renderText({ as.numeric(input$first_input) ** input$second_input })
  
  
}

shinyApp(ui, server)
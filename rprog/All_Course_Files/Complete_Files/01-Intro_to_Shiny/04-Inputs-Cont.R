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
)

server <- function(input, output) {
  
  observe({
    print( input$first_input )
  })
  
}

shinyApp(ui, server)
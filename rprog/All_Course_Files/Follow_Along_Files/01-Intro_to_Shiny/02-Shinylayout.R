library(shiny)
library(shinydashboard)
library(ggplot2)

# everything user sees
ui <- fluidPage(
  titlePanel("TItle Here"),
  "Hello World!",
  br(),
  
  selectInput(
    inputId="first_input",
    label = "Select a Number",
    choices = c(1:10),
    # multiple = TRUE, 
    # selected = 7
  ),

  numericInput(
    inputId="second_input",
    label = "Select a Number",
    value = 5,
    min = 1, max =20, step=1
  ),
  verbatimTextOutput( "power" )
)
  
  
  
server <- function(input, output) {
  # can't access input without observe
  observe({
    print(input$first_input)
    
  })
  make_power <- function(x,y) {as.numeric(x) ** as.numeric(y)}
  output$power <- renderText({ make_power(input$first_input, input$second_input) })
}

  

shinyApp(ui, server)


# ggplot( data = data.frame(x=1:10, y=(1:10) **5))















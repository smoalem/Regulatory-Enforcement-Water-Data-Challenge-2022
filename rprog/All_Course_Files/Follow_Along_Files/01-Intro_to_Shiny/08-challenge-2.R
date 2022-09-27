# Libraries ----
library(shiny)
library(shinydashboard)
library(ggplot2)


# UI ----
ui <- fluidPage(
  numericInput(
    inputId = "user_input",
    label = "Select a number",
    value = 1,
    min = 1, max = 10, step = 1
  ),

  plotOutput("plot")
)

server <- function(input, output) { 

  
  power_function <- function(x) {  (1:10) ** (1/x) }
  output$plot <- renderPlot({
    ggplot( data=data.frame(x = 1:10, y = 1:10 )) + 
      geom_point( aes(x,y = power_function(input$user_input))) + 
      geom_line(aes(x,y = power_function(input$user_input))) +
      ylab("y")
  })
}

shinyApp(ui, server)
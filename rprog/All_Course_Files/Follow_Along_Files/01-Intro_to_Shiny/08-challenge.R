# Libraries ----
library(shiny)
library(shinydashboard)
library(ggplot2)


# UI ----
ui <- fluidPage(
  numericInput(
    inputId = "first_input",
    label = "Select base number",
    value = 1,
    min = 1, max = 20, step = 1
  ),
  numericInput(
    inputId = "second_input",
    label = "Select inverse exponent",
    value = 2,
    min = 1, max = 20, step = 1
  ),
  verbatimTextOutput("power"),
  plotOutput("plot")
)

server <- function(input, output) { 
  observe({
    print( input$first_input  ** (1 / input$second_input))
  })
  
  make_power <- function(x, y) { as.numeric(x) ** as.numeric(y) }
  output$power <- renderText({ paste0("The answer of the inverse power is: ", make_power( input$first_input, (1/input$second_input) ))})
  output$plot <- renderPlot({
    ggplot( data=data.frame(x = 1:20, y = (1:20)** (1/ input$second_input)), aes(x,y)) + 
      geom_point() + 
      geom_line() + 
      geom_point(x = as.numeric(input$first_input),  y = make_power( input$first_input, (1/input$second_input) ), size = 4, color = "red")
  })
}

shinyApp(ui, server)

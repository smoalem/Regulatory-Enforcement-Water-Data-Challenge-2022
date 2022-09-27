# Libraries ----
library(shiny)
library(shinydashboard)
library(ggplot2)


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
  
  verbatimTextOutput("power"),
  
  plotOutput("plot"),
  
  
)

server <- function(input, output) {
  
  observe({
    print( as.numeric( input$first_input ) ** input$second_input)
  })
  
  make_power <- function(x, y) { as.numeric(x) ** as.numeric(y) }
  output$power <- renderText({ paste0("The answer of the power is: ", make_power( input$first_input, input$second_input ))})
  
  #output$power <- renderText({ as.numeric(input$first_input) ** input$second_input })
  
  output$plot <- renderPlot({
    ggplot( data=data.frame(x = 1:10, y = (1:10)** input$second_input), aes(x,y)) + 
      geom_point() + 
      geom_line() + 
      geom_point(x = as.numeric(input$first_input),  y = make_power(input$first_input, input$second_input), size = 4, color = "red")
  })
}

shinyApp(ui, server)


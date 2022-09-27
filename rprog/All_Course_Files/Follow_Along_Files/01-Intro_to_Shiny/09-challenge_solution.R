# Libraries ----
library(shiny)
library(shinydashboard)



# UI ----
ui <- fluidPage(
  
  h1("Welcome!"),
  p("Your challenge is to create a inverse power function plot!"),
  p("Your goal is to output a plot where X are the numbers 1:10 and Y are the numbers of X to the power of 1 divided by a user selected Input"),
  p("e.g. User Input: 2, Output: 1 ^ 1/2 , 2 ^ 1/2, 3 ^ 1/2, 4 ^ 1/2 ..."),
  
  numericInput(
    inputId = "user_input",
    label = "Select a Number",
    value = 1 ,
    min = 1, max = 10, step = 1
  ),
  
  plotOutput("plot")
  

)

server <- function(input, output) { 
  
  power_func <- function(x) {
    (1:10) ** (1/x)
  }

  output$plot <- renderPlot({
      ggplot( data.frame( x = 1:10, y = 1:10 ) ) +
      geom_point( aes( x, y = power_func(input$user_input)) ) +
      geom_line( aes( x, y = power_func(input$user_input)) ) +
      ylab("y")
  })
  
}

shinyApp(ui, server)








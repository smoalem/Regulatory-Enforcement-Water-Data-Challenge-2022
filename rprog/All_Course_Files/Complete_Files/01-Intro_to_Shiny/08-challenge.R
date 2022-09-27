# Libraries ----
library(shiny)
library(shinydashboard)



# UI ----
ui <- fluidPage(
  h1("Welcome!"),
  p("Your challenge is to create a inverse power function plot!"),
  p("Your goal is to output a plot where X are the numbers 1:10 and Y are the numbers of X to the power of 1 divided by a user selected Input"),
  p("e.g. User Input: 2, Output: 1 ^ 1/2 , 2 ^ 1/2, 3 ^ 1/2, 4 ^ 1/2 ..."),
  
)

server <- function(input, output) { 
  
}

shinyApp(ui, server)

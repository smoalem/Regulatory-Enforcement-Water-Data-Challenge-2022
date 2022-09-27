# Libraries ----
library(shiny)
library(shinydashboard)



# UI ----
ui <- fluidPage(
  title = "Welcome!",
  titlePanel("Title Here"),
  "Hello World",
  br(),
  p("My First Paragraph")
  )

server <- function(input, output) { 

  }

shinyApp(ui, server)
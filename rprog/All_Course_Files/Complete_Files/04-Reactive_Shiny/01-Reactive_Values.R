# Load packages ----
library(shiny)
library(quantmod)

# Source helpers ----
#source("helpers.R")

# User interface ----
ui <- fluidPage(
  titlePanel("stockVis"),

  sidebarLayout(
    sidebarPanel(
      helpText("Select a stock to examine.

        Information will be collected from Yahoo finance."),
      textInput("symb", "Symbol", "SPY"),

      dateRangeInput("dates",
                     "Date range",
                     start = "2013-01-01",
                     end = as.character(Sys.Date())),

      br(),
      br(),

      checkboxInput("log", "Plot y axis on log scale",
                    value = FALSE)),

    mainPanel(plotOutput("plot"))
  )
)

# Server logic
server <- function(input, output) {
  
  #observe( print( paste0("input$dates is: ", input$dates) ) )
  #isolate( print( paste0("input$dates using isolate is: ",input$dates)))
  
  
  values <- reactiveValues()
  
  observe( print( paste0("input is: ", input) ) ) 
  observe( print( paste0("values is: ", values) ) ) 
  
  values$a <- 3
  
  observe( print( paste0("values$a is: ", values$a) ) ) 
  


}

# Run the app
shinyApp(ui, server)

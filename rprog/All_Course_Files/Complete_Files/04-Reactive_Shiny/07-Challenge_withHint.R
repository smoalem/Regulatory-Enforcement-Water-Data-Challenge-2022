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
                    value = FALSE),
      
      actionButton(inputId = "go_bttn",
                   label = "Update Dates"
      ),
      # Add another Action Button Here
      actionButton(inputId = "clr_bttn",
                   label = "Clear"
      )
      
      ################################
      
    ),
    
    mainPanel(plotOutput("plot"))
  )
)

# Server logic
server <- function(input, output) {
  
  # Create Reactive Values Here 
  
  
  ##############################
  
  
  # Update Reactive Values Based on Buttons Here
  
  
  
  #########################################
  
  chart <- reactiveValues(value=0)
  observe(print(chart$value))
  observeEvent(input$go_bttn, {
    chart$value <- 1
  })
  
  observeEvent(input$clr_bttn, {
    chart$value <- 0
  })
  
  dataInput <- eventReactive(input$go_bttn,{
    #Sys.sleep(5)
    getSymbols(input$symb, src = "yahoo",
               from = input$dates[1],
               to = input$dates[2],
               auto.assign = FALSE)
  })
  
  
  output$plot <- renderPlot({
    
    # Create Logic to Display Chart Based on Reactive Values
    if(chart$value == 1) {
      chartSeries(dataInput(), theme = chartTheme("white"),
                  type = "line", log.scale = input$log, TA = NULL)
      
    } else {
      
    }
    
    
    
    #########################################################
    
  })
  
}

# Run the app
shinyApp(ui, server)

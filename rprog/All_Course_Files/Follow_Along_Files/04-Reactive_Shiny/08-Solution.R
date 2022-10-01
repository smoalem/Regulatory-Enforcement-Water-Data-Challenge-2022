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
      actionButton(inputId = "remove_bttn",
                   label = "Clear Chart",
                   class = "btn-danger"
      )
      
    ),
    
    mainPanel(plotOutput("plot"))
  )
)

# Server logic
server <- function(input, output) {
  
  chart <- reactiveValues(value=0)
  
  observe(print(chart$value))
  
  observeEvent(input$go_bttn, {
    chart$value <- 1
  })
  observeEvent(input$remove_bttn, {
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
    if( chart$value == 0 ){ } else {
    chartSeries(dataInput(), theme = chartTheme("white"),
                type = "line", log.scale = input$log, TA = NULL)
    }
    
  })
  
}

# Run the app
shinyApp(ui, server)

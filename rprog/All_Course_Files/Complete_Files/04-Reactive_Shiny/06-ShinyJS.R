# Load packages ----
library(shiny)
library(quantmod)
library(shinyjs)


# Source helpers ----
#source("helpers.R")

# User interface ----
ui <- fluidPage(
  
  titlePanel("stockVis"),

  sidebarLayout(
    
    
    sidebarPanel(
      
      useShinyjs(),
      
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
                   #class = "btn-success"
                  ),
      
      actionButton(inputId = "toggle_theme",
                   label = "Toggle Theme"),
      
      br(),
      br(),
      hr(),
      hidden(
      selectInput(inputId = "theme",
                  label = "Select a Theme",
                  choices = c("white", "black"),
                  selected = "white"
                  )
      )
      
      ),

    mainPanel( plotOutput("plot") )
  )
)

# Server logic
server <- function(input, output) {
  
  observeEvent(input$toggle_theme,{
    toggle("theme", anim = T, animType = "slide")
  })
  
  
  observe( print( paste0("Button is: ",input$go_bttn)) )

  
  observeEvent(input$go_bttn, {
    print( paste0("Button has been pushed!") )
  })

  dataInput <- eventReactive(input$go_bttn,{
     #Sys.sleep(5)
     getSymbols(input$symb, src = "yahoo",
                from = input$dates[1],
                to = input$dates[2],
                auto.assign = FALSE)
   })
  
  # dataInput <- reactive({
  #    #Sys.sleep(5)
  #    getSymbols(input$symb, src = "yahoo",
  #               from = input$dates[1],
  #               to = input$dates[2],
  #               auto.assign = FALSE)
  #  })
  
  # dataInput <- function(){
  #    #Sys.sleep(5)
  #    getSymbols(input$symb, src = "yahoo",
  #               from = input$dates[1],
  #               to = input$dates[2],
  #               auto.assign = FALSE)
  #  }

  output$plot <- renderPlot({

    chartSeries(dataInput(), theme = chartTheme(input$theme),
                type = "line", log.scale = input$log, TA = NULL )
    
  })

}

# Run the app
shinyApp(ui, server)

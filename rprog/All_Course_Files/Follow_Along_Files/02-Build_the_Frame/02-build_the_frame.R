# Libraries ----
library(shiny)
library(shinydashboard)
#library(shinyjs)
library(tidyverse)

# UI ----


ui <- dashboardPage(
  
  ####  Header ----
  dashboardHeader(
    title = "Covid 19 Country Comparison",
    titleWidth = 350
  ),
  
  #### Sidebar ----
  dashboardSidebar(
    width = 350,
    br(),
    h4("Select Your Inputs Here", style = "padding-left: 20px")
  ),
  
  #### Body ----
  dashboardBody(
    tabsetPanel(
      type = "tabs",
      id = "tab_selected",
      tabPanel(
        title = "Country View"
      )
   
    )
  )
)

# Server ----
server <- function(input, output){
  
}

shinyApp(ui, server)


# Other Stuff ----
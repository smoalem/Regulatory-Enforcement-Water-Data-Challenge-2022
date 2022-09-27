# Libraries ----
library(shiny)
library(shinydashboard)
library(shinyjs)
library(tidyverse)



# Initialize Data ----
dat <- readRDS(file = "app-data/subregion_agg.rds")



# UI ----
ui <- dashboardPage(
  
  #### Header ----
  dashboardHeader(
    title = "COVID-19 Country Comparison",
    titleWidth = 350
  ),
  #### Sidebar ----
  dashboardSidebar(
    
    width = 350,
    br(),
    h4("Select Your Inputs Here", style = "padding-left:20px")
    
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

server <- function(input, output) {
  
}

shinyApp(ui, server)













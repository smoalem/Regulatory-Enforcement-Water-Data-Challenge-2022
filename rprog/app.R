# Libs

library(shiny)
library(shinydashboard)
library(shinyjs)
library(tidyverse)
library(leaflet)
library(rgdal)
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%
library(jsonlite)
library(rgeos)
#



# #
# #
ui <- dashboardPage(
  dashboardHeader(
    title="TEST"
  ),
  dashboardSidebar(
    width = 350,
    uiOutput("sidebar")
    
  ),
  dashboardBody(

    tabsetPanel(
      type = "tabs",
      id = "tab_selected",
      tabPanel(
        title = "Map",
        leafletOutput("camap")

      ),
      tabPanel(
        title = "Data"
      ),
      tabPanel(
        title = "FAQ"

      )

    )
  )



)
# Server

server <- function(input, output) {
  ### dataframe load
  # map_switch <- reactiveValue()
  res <- readOGR(dsn = "./rprog/water_system_with_scores_updated_3.geojson", layer="water_system_with_scores_updated_3")
### Render UI
  dropdown_input <- reactive({
    input$complianceOverageInput
  })


  output$complianceOverageOutput <- renderUI({
    selectInput(
      inputId = "complianceOverageInput", 
      label = strong("Select Score Type:", style = "font-family: 'arial'; font-si28pt"),
      choices =  c("Compliance Score", "Overage Score"),
      selected = "Compliance Score"
    )
  })
  
  output$sidebar <- renderUI({
    if( input$tab_selected == "Map"){
      div(
        uiOutput("complianceOverageOutput")
      
      )
    } 
    
  })
  
 

   
    output$camap <- renderLeaflet(
      {
        if(dropdown_input() == "Compliance Score") { 
          binpal <-   colorBin("Blues", res$red_int, 9, pretty = FALSE) 
          map <- leaflet(res) %>%
            addTiles() %>%
            addPolygons(
              stroke = FALSE,
              smoothFactor = 0.2,
              fillOpacity = 0.9,
              dashArray = "3",
              color = ~binpal(red_int),
              popup = ~paste0(
                "<b> Water System Number: <b>",
                WATER_SYST,
                "<br>",
                "<b> Water System Name: <b>",
                WATER_SY_1
              )
            ) %>%
            addLegend("bottomright", pal = binpal, values = ~red_int,
                      title = "Legend",
                      opacity = 1
            ) %>%
            setView(lng = -119.417931, lat = 36.778259, zoom = 5)
          
        }   else { 
          binpal <-   colorBin("Greens", res$red_int, 9, pretty = FALSE) 
          map <- leaflet(res) %>%
            addTiles() %>%
            addPolygons(
              stroke = FALSE,
              smoothFactor = 0.2,
              fillOpacity = 0.9,
              dashArray = "3",
              color = ~binpal(red_int),
              popup = ~paste0(
                "<b> Water System Number: <b>",
                WATER_SYST,
                "<br>",
                "<b> Water System Name: <b>",
                WATER_SY_1
              )
            ) %>%
            addLegend("bottomright", pal = binpal, values = ~red_int,
                      title = "Legend",
                      opacity = 1
            ) %>%
            setView(lng = -119.417931, lat = 36.778259, zoom = 5)
          
        } 
        map
        
      }
    )
      
      
      

}


#
# # Shiny App
#
shinyApp(ui, server)
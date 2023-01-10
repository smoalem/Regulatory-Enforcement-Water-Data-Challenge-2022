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
library(sf)
#



# #
# #
ui <- dashboardPage(
  dashboardHeader(
    title="Water System Regulatory Compliance Review Map"
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
        title = "Data Source",
        tags$a(href="https://github.com/smoalem/Regulatory-Enforcement-Water-Data-Challenge-2022", "Reulatory Enforcement Github Page"),
        br(),
        "Please post your questions to sarmad.moalem@vapyranalytics.com"

      )

    )
  )



)
# Server

server <- function(input, output) {
  ### dataframe load
  # map_switch <- reactiveValue()
  res <- st_read(dsn = "./water_system_with_scores_updated_3.geojson", layer="water_system_with_scores_updated_3")
### Render UI
  dropdown_input <- reactive({
    input$complianceOverageInput
  })


  output$complianceOverageOutput <- renderUI({
    selectInput(
      inputId = "complianceOverageInput", 
      label = strong("Select Percentile Type:", style = "font-family: 'arial'; font-si28pt"),
      choices =  c("Compliance Percentile", "Overage Percentile"),
      selected = "Compliance Percentile"
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
        if(dropdown_input() == "Compliance Percentile") { 
          binpal <-   colorBin("Reds", res$ave_score_red_lean_percentile, 9, pretty = FALSE) 
          map <- leaflet(res) %>%
            addTiles() %>%
            addPolygons(
              stroke = TRUE,
              smoothFactor = 0.2,
              fillOpacity = 0.9,
              dashArray = "3",
              color = "black",
              fillColor = ~binpal(ave_score_red_lean_percentile),
              opacity = 1,
              popup = ~paste0(
                "<b> Water System Number: <b>",
                WATER_SYST,
                "<br>",
                "<b> Water System Name: <b>",
                WATER_SY_1,
                "<br>",
                "<b> Compliance Percentile: <b>",
                ave_score_red_lean_percentile,
                "<br>",
                "<b> Overage Percentile: <b>",
                overage_percentile
              )
            ) %>%
            addLegend("bottomright", pal = binpal, values = ~ave_score_red_lean_percentile,
                      title = "Compliance Percentile",
                      opacity = 1
            ) %>%
            setView(lng = -119.417931, lat = 36.778259, zoom = 5)
          
        }   else { 
          binpal <-   colorBin("Purples", res$overage_percentile, 5, pretty = FALSE) 
          map <- leaflet(res) %>%
            addTiles() %>%
            addPolygons(
              stroke = TRUE,
              smoothFactor = 0.2,
              fillOpacity = 0.9,
              dashArray = "3",
              opacity = 1,
              color = "black",
              fillColor = ~binpal(overage_percentile),
              popup = ~paste0(
                "<b> Water System Number: <b>",
                WATER_SYST,
                "<br>",
                "<b> Water System Name: <b>",
                WATER_SY_1,
                "<br>",
                "<b> Compliance Percentile: <b>",
                ave_score_red_lean_percentile,
                "<br>",
                "<b> Overage Percentile: <b>",
                overage_percentile
              )
            ) %>%
            addLegend("bottomright", pal = binpal, values = ~overage_percentile,
                      title = "Overage Percentile",
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
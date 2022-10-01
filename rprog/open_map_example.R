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

url <- "C:/Users/jheaf/Documents/Regulatory-Enforcement-Water-Data-Challenge-2022/removed_neg_scores.geojson"
res <- readOGR(dsn = url, layer="removed_neg_scores")
# centers <- data.frame(gCentroid(res, byid = TRUE))
# res$lng <- centers$x
# res$lat <- centers$y
pal <- colorNumeric(
  palette = "Blues",
  domain = res$red_int
)
# 

binpal <- colorBin("Blues", res$red_int, 10, pretty = FALSE)


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
### Map 
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


  output$camap <- renderLeaflet(map)
### Render UI
  output$complianceOverage <- renderUI({
    selectInput(
      inputId = "complianceOverage", 
      label = strong("Select Score Type:", style = "font-family: 'arial'; font-si28pt"),
      choices =  c("Compliance Score", "Overage Score"),
      selected = choices[1]
    )
  })
  
  output$sidebar <- renderUI({
    if( input$tab_selected == "Map"){
      div(
        uiOutput("complianceOverage")
      )
    } 
    
  })

}


#
# # Shiny App
#
shinyApp(ui, server)
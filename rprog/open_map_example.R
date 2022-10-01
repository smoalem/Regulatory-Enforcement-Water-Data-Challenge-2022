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

url <- "C:/Users/jheaf/Documents/Regulatory-Enforcement-Water-Data-Challenge-2022/new_export_test.geojson"
res <- readOGR(dsn = url, layer="new_export_test")
centers <- data.frame(gCentroid(res, byid = TRUE))
res$lng <- centers$x
res$lat <- centers$y
regular_data <- data.frame(res[, c(2,15, 35)])

#
# #
ui <- dashboardPage(
  dashboardHeader(
    title="TEST"
  ),
  dashboardSidebar(
    width = 350
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
# #
# # # Server
#
server <- function(input, output) {
  popup <- 
  map <- leaflet(res) %>%
      addTiles() %>%
      addPolygons(weight=2, col = "red", popup = ~paste0(
        "<b> Water System Number: <b>", 
        WATER_SYST,
        "<br>",
        "<b> Water System Name: <b>", 
        WATER_SY_1
        ), fillColor = "yellow") %>%
      setView(lng = -119.417931, lat = 36.778259, zoom = 5)


  output$camap <- renderLeaflet(map)


}


#
# # Shiny App
#
shinyApp(ui, server)
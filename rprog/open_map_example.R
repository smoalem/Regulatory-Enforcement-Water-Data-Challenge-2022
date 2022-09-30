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

url <- "C:/Users/user/Documents/GitHub/Regulatory-Enforcement-Water-Data-Challenge-2022/new_export_test.geojson"
 

res <- readOGR(dsn = url, layer="new_export_test")
poly <- res[, 0]@polygons
centers <- data.frame(gCentroid(res, byid = TRUE))
regular_data <- data.frame(res[, c(2,15, 35)])

# #
# ui <- dashboardPage(
#   dashboardHeader(
#     title="TEST"
#   ),
#   dashboardSidebar(
#     width = 350
#   ),
#   dashboardBody(
# 
#     tabsetPanel(
#       type = "tabs",
#       id = "tab_selected",
#       tabPanel(
#         title = "Map",
#         leafletOutput("camap")
# 
#       ),
#       tabPanel(
#         title = "Data"
#       ),
#       tabPanel(
#         title = "FAQ"
# 
#       )
# 
#     )
#   )
# 
# 
# 
# )
# #
# # # Server
# #
# server <- function(input, output) {
# 
#   map <- leaflet(poly) %>%
#       addTiles() %>%
#       addPolygons(weight=2, col = "red", fillColor = "yellow") %>%
#       setView(lng = -119.417931, lat = 36.778259, zoom = 5)
# 
# 
#   output$camap <- renderLeaflet(map)
# # #
# # #
# # # }
# #
# #
# #
# # # Shiny App
# #
# shinyApp(ui, server)
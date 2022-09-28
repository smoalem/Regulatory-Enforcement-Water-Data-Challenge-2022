library(shiny)
library(shinydashboard)
library(shinyjs)
library(tidyverse)
library(leaflet)
library(rgdal)
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%
library(jsonlite)

# url <- "C:/Users/jheaf/Dropbox/Water Data Challenge 2022/GIS/water_system_with_scores.geojson"
url <- "C:/Users/user/Documents/GitHub/Regulatory-Enforcement-Water-Data-Challenge-2022/new_export_test.geojson"

res <- readOGR(dsn = url)
# geojson <- jsonLite::readLines(url) %>%
#   paste(collapse = "\n") %>%
#   fromJSON(simplifyVector = FALSE)



leaflet(res) %>% 
  addTiles() %>%
  addPolylines(fillColor = "red") %>%
  addPolygons(fillColor = "yellow") %>%
  setView(lng = -119.417931, lat = 36.778259, zoom = 5)

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
url <- "C:/Users/jheaf/Dropbox/Water Data Challenge 2022/new_export_test.geojson"

res <- readOGR(dsn = url)
# geojson <- jsonLite::readLines(url) %>%
#   paste(collapse = "\n") %>%
#   fromJSON(simplifyVector = FALSE)



leaflet() %>% 
  addTiles() %>%
  setView(lng = -117.3884190, lat = 34.1776714, zoom = 10) %>% 
  addGeoJSON(geojson = res)


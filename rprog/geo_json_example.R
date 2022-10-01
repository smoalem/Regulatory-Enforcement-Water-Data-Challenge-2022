library(shiny)
library(shinydashboard)
library(shinyjs)
library(tidyverse)
library(leaflet)
library(rgdal)

# From http://eric.clst.org/Stuff/USGeoJSON and
# https://en.wikipedia.org/wiki/List_of_United_States_counties_and_county_equivalents
nycounties <- rgdal::readOGR("https://rstudio.github.io/leaflet/json/nycounties.geojson") 

# Or use the geojsonio equivalent:
# nycounties <- geojsonio::geojson_read("https://rstudio.github.io/leaflet/json/nycounties.geojson", what = "sp") 

pal <- colorNumeric("viridis", NULL)

leaflet(nycounties) %>%
  addTiles() %>%
  addPolygons(stroke = FALSE, smoothFactor = 0.3, fillOpacity = 1,
              fillColor = ~pal(log10(pop)),
              label = ~paste0(county, ": ", formatC(pop, big.mark = ","))) %>%
  addLegend(pal = pal, values = ~log10(pop), opacity = 1.0,
            labFormat = labelFormat(transform = function(x) round(10^x)))


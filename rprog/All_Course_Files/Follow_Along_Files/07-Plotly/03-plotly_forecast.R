rm(list=ls())
# Libraries ----
library(tidyverse)
library(tidyquant)
library(scales)
library(forecast)
library(plotly)


dat <- readRDS(file = "Steps/app-data/subregion_agg.rds")

dat <-
  dat %>%
  select( !subregion1_name ) %>%
  filter( country_name %in% c("Canada","France") & date >= "2020-06-01" & date <= "2020-12-31") %>%
  group_by(country_name,date) %>%
  summarise_all(sum) %>%
  select( country_name, date, "new_confirmed") %>%
  set_names( c("country_name","date","metric")) %>% 
  arrange(date)


# Trend line
the_metric <- "New Confirmed"
plt <- plot_ly (data = dat, x=~date, color=~country_name, text = ~country_name)
plt <- plt %>% add_trace(y=~metric, type='scatter', mode='lines+markers', #fill = 'tonexty',
                         hovertemplate = paste(
                           paste0('<extra></extra>Country Name: %{text}\n',the_metric,': %{y}\nDate: %{x} ')
                         ))
#plt <- plt %>% layout(hovermode = 'x unified')

highlight(plt)













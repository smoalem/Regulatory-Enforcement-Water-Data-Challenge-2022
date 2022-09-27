rm(list=ls())
# Libraries ----
library(tidyverse)
library(tidyquant)
library(scales)
library(forecast)

num_forecasts = 6

dat <- readRDS(file = "app-data/subregion_agg.rds")

dat <-
  dat %>%
  select( !subregion1_name ) %>%
  filter( country_name %in% c("Canada") & date >= "2020-06-01" & date <= "2020-12-31") %>%
  group_by(country_name,date) %>%
  summarise_all(sum) %>%
  select( country_name, date, "new_confirmed") %>%
  set_names( c("country_name","date","metric")) %>% 
  arrange(date)


create_forecast <- function(dat, num_forecasts){
  name_country <- unique(dat$country_name)
  auto_forecast <-  forecast(auto.arima(dat$metric),num_forecasts)$mean
  max_date <- max(dat$date)
  new_dates <- max_date + c(1:num_forecasts)
  new_forecast <- tibble( country_name = name_country, date = new_dates , metric = as.vector(auto_forecast), forecast = 1 )
  return(new_forecast)
}

create_forecast(dat, num_forecasts)

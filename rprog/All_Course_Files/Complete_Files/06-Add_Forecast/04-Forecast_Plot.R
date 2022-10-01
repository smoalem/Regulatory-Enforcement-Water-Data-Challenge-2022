rm(list=ls())
# Libraries ----
library(tidyverse)
library(tidyquant)
library(scales)
library(forecast)


dat <- readRDS(file = "app-data/subregion_agg.rds")

dat <-
  dat %>%
  select( !subregion1_name ) %>%
  filter( country_name %in% c("Canada","France") & date >= "2020-06-01" & date <= "2020-12-31") %>%
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

create_forecast(dat, 6)


predictions_by_country <- function(){
  dat %>%
    group_by(country_name) %>%
    group_map(~ create_forecast(.x, num_forecasts=6), .keep=T )
}

predictions_by_country()


forecast_data <- function(){
  unforecasted_data <- dat
  unforecasted_data$forecast <- 0
  forecasted_data <- predictions_by_country()
  forecasted_data <- do.call(rbind,forecasted_data)
  rbind(unforecasted_data,forecasted_data) 
}

foo <- forecast_data()
foo


ggplot( data = forecast_data() %>% filter(forecast==0), aes(y = metric, x = date, color=country_name) ) +
  geom_line(size = 1.5) +
  #geom_ma(n=ma_days,size=1) +
  geom_line(data = forecast_data() %>% filter(forecast==1),size = 2.5,linetype=7,alpha=0.25) +
  scale_y_continuous( label=comma) +
  theme_bw() +
  geom_vline(xintercept= forecast_data() %>% filter(forecast==0) %>% pull(date) %>% max, linetype="dotdash",size=0.5)









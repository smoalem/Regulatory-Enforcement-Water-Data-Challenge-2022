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


create_forecast <- function(dat, num_forecasts){
  name_country <- unique(dat$country_name)
  auto_forecast <-  forecast(auto.arima(dat$metric),num_forecasts)$mean
  max_date <- max(dat$date)
  new_dates <- max_date + c(1:num_forecasts)
  new_forecast <- tibble( country_name = name_country, date = new_dates , metric = as.vector(auto_forecast), forecast = 1 )
  return(new_forecast)
}

predictions_by_country <- function(){
  dat %>%
    group_by(country_name) %>%
    group_map(~ create_forecast(.x, num_forecasts=6), .keep=T )
}

forecast_data <- function(){
  unforecasted_data <- dat
  unforecasted_data$forecast <- 0
  forecasted_data <- predictions_by_country()
  forecasted_data <- do.call(rbind,forecasted_data)
  rbind(unforecasted_data,forecasted_data) 
}

forcastData <- forecast_data()
forcastData

forcastData2 <- rbind( 
  forcastData %>% filter(forecast==0) %>%  filter(date == max(date)),
  forcastData %>% filter(forecast==1)
)


plt <- plot_ly (data = forcastData %>% filter(forecast == 0), x=~date, color =~country_name, text=~country_name)
plt <- plt %>% add_trace(y=~metric, type='scatter', mode='lines+markers', marker = list(size = 10), 
                         hovertemplate = paste(
                           paste0('<extra></extra>Country Name: %{text}\n',the_metric,': %{y}\nDate: %{x} ')))
plt <- plt %>% add_trace(data = forcastData2 , y=~metric, x=~date, color = ~country_name,
                         type='scatter', mode="lines" , showlegend=F, line=list(color="grey", dash="dot"), #,width=4
                         hovertemplate = paste(
                           paste0("<extra>Forecast</extra>Country Name: %{text}\n",the_metric,': %{y}\nDate: %{x}')) )
plt <- plt %>%  add_ribbons(x = ~date,
                            ymin =~metric*0.8, 
                            ymax =~metric*1.2,
                            color = I("gray95"),
                            showlegend=F,
                            hoverinfo="skip")


highlight(plt)









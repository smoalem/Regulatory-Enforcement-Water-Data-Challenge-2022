rm(list=ls()) # clean stuff from environment

library(tidyverse)
library(data.table)

dat <- fread(file = "Data/covid_data.csv")



dat_country <- dat[ , lapply(.SD, sum, na.rm=T), by=c("country_name", "date"), .SDcols=c(12:25)]
dat_subregion <- dat[, lapply(.SD,sum,na.rm=T), by=c("country_name", "subregion1_name", "date"), .SDcols=c(12:25)]

dat_country_t <- tibble(dat_country)
dat_subregion_t <- tibble(dat_subregion)
# 
saveRDS(dat_country_t, file="country_agg.rds")
saveRDS(dat_subregion_t, file="subregion_agg.rds")
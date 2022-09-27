
dat <- readRDS(file = "app-data/subregion_agg.rds")



dat %>%
  filter( country_name == "Canada" & date >= '2020-06-01' & date <= '2020-06-06' ) %>%
  select( !country_name ) %>%
  group_by(subregion1_name,date) %>%
  summarise_all(sum) %>%
  select( subregion1_name, date, "new_confirmed") %>%
  filter( subregion1_name != "" & subregion1_name %in% "Ontario" ) %>%
  set_names( c("subregion1_name","date","metric")) %>% 
  arrange(date)

SELECT
date,country_code,country_name,subregion1_code,
subregion1_name,subregion2_code,subregion2_name,aggregation_level,new_confirmed,new_deceased,new_recovered, new_tested,cumulative_confirmed,            
cumulative_deceased,cumulative_recovered,cumulative_tested,new_hospitalized_patients,         
new_intensive_care_patients,new_ventilator_patients,cumulative_hospitalized_patients,cumulative_intensive_care_patients,
cumulative_ventilator_patients,current_hospitalized_patients,current_intensive_care_patients,current_ventilator_patients                  
FROM `bigquery-public-data.covid19_open_data.covid19_open_data`
WHERE date >= '2020-01-01' AND date <= '2021-02-25'

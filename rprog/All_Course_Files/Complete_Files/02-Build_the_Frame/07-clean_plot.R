# Libraries ----
library(shiny)
library(shinydashboard)
library(shinyjs)
library(tidyverse)
library(scales)



# Initialize Data ----
dat <- readRDS(file = "app-data/subregion_agg.rds")

metric_choices <- colnames(dat)[4:ncol(dat)]
metric_names <- gsub("_", " ", metric_choices)
metric_names <- paste0(toupper(substr(metric_names,1,1)), substr(metric_names,2,nchar(metric_names)))
metric_list <- as.list(metric_choices)
names(metric_list) <- metric_names


# UI ----
ui <- dashboardPage(
  
  #theme = bs_theme(version = 4, bootswatch = "minty"),
  skin = "red",
  
  #### Header ----
  dashboardHeader(
    title = "COVID-19 Country Comparison",
    titleWidth = 350
  ),
  #### Sidebar ----
  dashboardSidebar(
    
    width = 350,
    br(),
    h4("Select Your Inputs Here", style = "padding-left:20px"),
    
    # metric Input ----
    selectInput(
      inputId = "metric", 
      label = strong("Select Metric:", style = "font-family: 'arial'; font-si28pt"),
      choices =  metric_list,
      selected = metric_list[1]
    ),
    
    # country Input ----
    selectInput(
      inputId = "country", 
      multiple = TRUE,
      label = strong("Select Countries to Compare:", style = "font-family: 'arial'; font-si28pt"),
      choices = sort(unique(dat$country_name)),
      selected = c("United States of America","France","Canada")
    ),
    
    # date_range_country Input ----
    dateRangeInput(
      inputId = "date_range_country",
      label = "Select Date Range:",
      start = "2020-01-01",
      end   = "2020-12-31"
    )
    
  ),
  #### Body ----
  dashboardBody(
    tabsetPanel(
      type = "tabs",
      id = "tab_selected",
      tabPanel(
        title = "Country View",
        plotOutput("plot_data_country")
      )
    )
  )
)

server <- function(input, output) {
  
  
  # _________________________ -----
  # 1 - Data Cleaning Functions ----
  # 01.A clean_data_country() ----
  clean_data_country <- reactive({
    clean_dat <- dat %>%
      select( !subregion1_name ) %>%
      filter( country_name %in% input$country & date >= input$date_range_country[1] & date <= input$date_range_country[2]) %>%
      group_by(country_name,date) %>%
      summarise_all(sum) %>%
      select( country_name, date, input$metric) %>%
      set_names( c("country_name","date","metric")) %>% 
      arrange(date)
  })
  
  # _________________________ -----
  
  # 2 - Plotting Data ----
  
  # 02.A plot_data_country ----
  output$plot_data_country <- renderPlot({
    ggplot( data = clean_data_country(), aes(y = metric, x = date, color=country_name) ) +
      geom_line(size = 1.5) +
      ylab( metric_names[which(metric_choices == input$metric)] ) +
      xlab( "Date" ) +
      labs(color="Country Name") +
      scale_y_continuous( label = comma ) +
      scale_x_date( date_breaks = "1 month", date_labels =  "%b %Y" ) +
      # theme(
      #      #panel.background = element_blank(),
      #       axis.text.x = element_text(size=16),
      #       axis.text.y=element_text(size=16),
      #       axis.title.x = element_text(size=18),
      #       axis.title.y = element_text(size=18),
      #       strip.text = element_text(size=25)
      # ) +
      ggtitle( metric_names[which(metric_choices == input$metric)] )
  })
  
}

shinyApp(ui, server)
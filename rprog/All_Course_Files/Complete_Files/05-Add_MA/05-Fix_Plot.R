# Libraries ----
library(shiny)
library(shinydashboard)
library(shinyjs)
library(tidyverse)
library(scales)
library(tidyquant)


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
    
    shinyjs::useShinyjs(),
    
    width = 350,
    br(),
    h4("Select Your Inputs Here", style = "padding-left:20px"),
    
    uiOutput("sidebar")
    
  ),
  #### Body ----
  dashboardBody(
    tabsetPanel(
      type = "tabs",
      id = "tab_selected",
      tabPanel(
        title = "Country View",
        plotOutput("country_plot")
      ),
      tabPanel(
        # 2 - Regional View ----
        title = "Regional View",
        plotOutput("regional_plot")
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
  
  # 01.B clean_data_regional() ----
  clean_data_regional <- reactive({
    dat %>%
      filter( country_name %in% input$country_single & date >= input$date_range_regional[1] & date <= input$date_range_regional[2]) %>%
      select( !country_name ) %>%
      group_by(subregion1_name,date) %>%
      summarise_all(sum) %>%
      select( subregion1_name, date, input$metric) %>%
      filter( subregion1_name != "" & subregion1_name %in% input$subregion ) %>%
      set_names( c("subregion1_name","date","metric")) %>% 
      arrange(date)
  })
  
  # _________________________ -----
  
  # 2 - Plotting Data ----
  
  # 02.A Functions ----
  plot_data_country <- function(){
    ma_days <- ifelse( input$moving_average == T , ma_days(), 0 )
    ggplot( data=clean_data_country(), aes(y = metric, x = date, color=country_name) ) +
      geom_line(size = 1.5) +
      geom_ma(n=ma_days,size=1) +
      ylab( metric_names[which(metric_choices == input$metric)] ) +
      xlab( "Date" ) +
      labs(color="Country Name") +
      scale_y_continuous( label=comma ) +
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
  }
  
  plot_data_region <- function(){
    date_breaks <- ifelse( round(time_diff()/10) >= 1 , round(time_diff()/10), 1 )
      ggplot( data = clean_data_regional() %>% filter(subregion1_name != ""), aes(y = metric, x = date, fill=subregion1_name) ) +
        geom_bar(stat="identity") +
        ylab( metric_names[which(metric_choices == input$metric)] ) +
        xlab( "Date" ) +
        labs(fill="Sub Region Name") +
        scale_x_date( date_breaks = paste0(date_breaks,' day'), date_labels =  "%b-%d" ) +
        # theme(
        #   #panel.background = element_blank(),
        #   axis.text.x = element_text(size=16),
        #   axis.text.y=element_text(size=16),
        #   axis.title.x = element_text(size=18),
        #   axis.title.y = element_text(size=18),
        #   strip.text = element_text(size=25)
        # )  +
        ggtitle( metric_names[which(metric_choices == input$metric)] )
  }
  
  # 02.B Render Plots ----
  output$country_plot <- renderPlot({
    req( input$country ) 
    plot_data_country()
  })
  output$regional_plot <- renderPlot({
    req( input$country_single)
    plot_data_region()
  })
  
  
  # _________________________ -----
  # 3 Buttons, Toggles, and Functions ----
  
  # 03.A ma_days (days in MA) ----
  ma_days <- eventReactive(input$moving_average_bttn,{
    req(input$moving_average_days)
    input$moving_average_days
  },ignoreNULL = FALSE)
  
  # 03.B Toggle MA ----
  observeEvent(input$moving_average, {
    if( input$moving_average == TRUE )
      shinyjs::show(id = "moving_average_days", anim = TRUE, animType = "slide") 
    else {
      shinyjs::hide(id = "moving_average_days", anim = TRUE, animType = "fade")
    }
  })
  
  # 03.C time_diff (Time Difference Number of Days Selected) ----
  time_diff <- reactive({
    req(input$date_range_regional)
    ( as.Date(input$date_range_regional[2]) - as.Date(input$date_range_regional[1]) )[[1]]
  })
  
  
  # _________________________ -----
  
  # 4 SELECT Inputs ----
  
  # 04.A metric ----
  output$metric <- renderUI({
    selectInput(
      inputId = "metric", 
      label = strong("Select Metric:", style = "font-family: 'arial'; font-si28pt"),
      choices =  metric_list,
      selected = metric_list[1]
    )
  })
  # 04.B country ----
  output$country <- renderUI({
    selectInput(
      inputId = "country", 
      multiple = TRUE,
      label = strong("Select Countries to Compare:", style = "font-family: 'arial'; font-si28pt"),
      choices = unique(dat$country_name),
      selected = c("United States of America","France","Canada")
    )
  })
  
  
  # 04.C date_range_country ----
  output$date_range_country <- renderUI({
    dateRangeInput(
      inputId = "date_range_country",
      label = "Select Date Range:",
      start = "2020-01-01",
      end   = "2020-12-31"
    )
  })
  
  # 04.D date_range_regional ----
  output$date_range_regional <- renderUI({
    dateRangeInput(
      inputId = "date_range_regional",
      label = "Select Date Range:",
      # start = initial_start(),
      # end = initial_end()
      start = "2020-05-05",
      end   = "2020-05-11"
    )
  })
  
  # 04.E moving_average ----
  output$moving_average <- renderUI({
    checkboxInput(
      inputId = "moving_average",
      label = div("Include Moving Average", style = "font-size: 12pt"),
      #style = "font-size: 28pt",
      value = FALSE
    )
  })
  
  # 04.F moving_average_days ----
  output$moving_average_days <- renderUI({
    div(
      numericInput(
        inputId = "moving_average_days",
        label = "Number of Days for Moving Average",
        value = 5,
        min = 0,
        #max = 500,
        step = 1
      ),
      actionButton(inputId = "moving_average_bttn",
                   label = "Update MA",
                   class = "btn-success"
      )
    )
  })  
  
  # 04.G country_single ----
  output$country_single <- renderUI({
    selectInput(
      inputId = "country_single", 
      multiple = FALSE,
      label = strong("Select a Country:", style = "font-family: 'arial'; font-si28pt"),
      choices = unique(dat$country_name),
      selected = c("Canada")
    )
  })
  
  # 04.H subregion ----
  get_unique_subregions <- reactive({
    req( input$country_single )
    dat %>% filter(country_name==input$country_single & subregion1_name != "") %>%
      select(subregion1_name) %>% unique %>% pull
  })
  output$subregion <- renderUI({
    selectInput(
      inputId = "subregion",
      multiple = TRUE,
      label = "Select a Subregion:",
      choices = get_unique_subregions(),
      selected = (get_unique_subregions())[1]
    )
  })
  
  # 5 UI Sidebar Output ----
  output$sidebar <- renderUI({
    if( input$tab_selected == "Country View"){
      div(
        uiOutput("metric"),
        uiOutput("country"),
        uiOutput("date_range_country"),
        uiOutput("moving_average"),
        uiOutput("moving_average_days") %>% hidden()
      )
    } else if ( input$tab_selected == "Regional View" ) {
      div(
        uiOutput("metric"),
        uiOutput("country_single"),
        uiOutput("subregion"),
        uiOutput("date_range_regional")
      )
    }
  })
  
}

shinyApp(ui, server)
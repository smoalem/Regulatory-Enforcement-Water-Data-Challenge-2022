# Libraries ----
library(shiny)
library(shinydashboard)
library(shinyjs)
library(tidyverse)
library(scales)
library(tidyquant)
library(forecast)
library(plotly)

source("app-data/functions.R")
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
        plotlyOutput("country_plot"),
        uiOutput("forecast_panel")
      ),
      tabPanel(
        # 2 - Regional View ----
        title = "Regional View",
        plotlyOutput("regional_plot")
      )
    )
  )
)

server <- function(input, output) {
  
  make_forecast <- reactiveValues(value=0)
  
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
  plot_data_country <- function(data){
    ifelse( input$moving_average == T & !is.null(input$moving_average_days),
            dat_ma <- clean_data_country() %>%
              group_by(country_name) %>%
              mutate(ma2=rollapply(metric,ma_days(),mean,align='right',fill=NA))
            ,
            dat_ma <- clean_data_country()
    )
    f <- list(family = "Courier New, monospace",size = 18,color = "black")
    x <- list(title = "Date",titlefont = f)
    y <- list( title = "Metric",titlefont = f)
    
    plt <- plot_ly (data = dat_ma, x=~date, color=~country_name, text = ~country_name)
    plt <- plt %>% add_trace(y=~metric, type='scatter', mode='lines+markers', #fill = 'tonexty',
                             hovertemplate = paste(
                               paste0('<extra>Actuals</extra>Country Name: %{text}\n',input$metric,': %{y}\nDate: %{x} ')
                             ))
    if( input$moving_average == T & !is.null(input$moving_average_days) ){
      plt <- plt %>% add_trace(y=~ma2, type='scatter', mode='lines', line=list(dash="dot"), showlegend=F,
                               hovertemplate = paste(
                                 paste0("<extra>Moving Average</extra>Country Name: %{text}\n",input$metric,': %{y}\nDate: %{x}')) )
    }
    plt <- layout(plt, title = '', yaxis = y, xaxis = x)
    highlight(plt)
  }
  
  plot_data_country_forecast <- function(data){
    ma_days <- ifelse( input$moving_average == T , ma_days(), 0 )
    ggplot( data = forecast_data() %>% filter(forecast==0), aes(y = metric, x = date, color=country_name) ) +
      geom_line(size = 1.5) +
      geom_ma(n=ma_days,size=1) +
      geom_line(data = forecast_data() %>% filter(forecast==1),size = 2.5,linetype=7,alpha=0.25) +
      ylab( metric_names[which(metric_choices == input$metric)] ) +
      xlab("Date") +
      scale_x_date( date_breaks = "1 month", date_labels =  "%b %Y" ) +
      scale_y_continuous( label=comma) +
      theme_bw() +
      geom_vline(xintercept= forecast_data() %>% filter(forecast==0) %>% pull(date) %>% max, linetype="dotdash",size=0.5)
  }
  
  plot_data_region <- function(data){
    date_breaks <- ifelse( round(time_diff()/10) >= 1 , round(time_diff()/10), 1 )
    ggplotly(
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
    )
  }
  
  # 02.B Render Plots ----
  output$country_plot <- renderPlotly({
    req( input$country ) 
    ifelse( make_forecast$value == 0, return(plot_data_country( clean_data_country() )), return(plot_data_country_forecast( clean_data_country() )) )
  })
  output$regional_plot <- renderPlotly({
    plot_data_region( clean_data_regional() )
  })
  
  # _________________________ -----
  # 3 Buttons, Toggles, and Functions ----
  
  # 03.A ma_days (days in MA) ----
  ma_days <- eventReactive(input$moving_average_bttn,{
    req(input$moving_average_days)
    input$moving_average_days
  },ignoreNULL = FALSE)
  
  # 03.B forecast_bttn ----
  observeEvent(input$forecast_bttn, {
    make_forecast$value <- 1
  })
  # 03.C remove_forecast_bttn ----
  observeEvent(input$remove_forecast_bttn, {
    make_forecast$value <- 0
  })
  forecast_days <- eventReactive(input$forecast_bttn,{
    input$forecast
  })
  
  # 03.D Toggle MA ----
  observeEvent(input$moving_average, {
    if( input$moving_average == TRUE )
      shinyjs::show(id = "moving_average_days", anim = TRUE, animType = "slide") 
    else {
      shinyjs::hide(id = "moving_average_days", anim = TRUE, animType = "fade")
    }
  })
  
  # 03.E time_diff (Time Difference Number of Days Selected) ----
  time_diff <- reactive({
    req(input$date_range_regional)
    ( as.Date(input$date_range_regional[2]) - as.Date(input$date_range_regional[1]) )[[1]]
  })
  
  # 03.F predictions_by_country ----
  predictions_by_country <- reactive({
    clean_data_country() %>%
      group_by(country_name) %>%
      group_map(~ create_forecast(.x, num_forecasts=forecast_days()), .keep=T )
  })
  # 03.G forecast_data ----
  forecast_data <- reactive({
    unforecasted_data <- clean_data_country()
    unforecasted_data$forecast <- 0
    forecasted_data <- predictions_by_country()
    forecasted_data <- do.call(rbind,forecasted_data)
    rbind(unforecasted_data,forecasted_data)
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
  
  # 04.I forecast ----
  output$forecast <- renderUI({
    numericInput(
      inputId = "forecast",
      label = "Number of Days to Forecast",
      value = 20, min = 0, max = 100, step = 1
    )
  })
  # 04.J forecast_bttn ----
  output$forecast_bttn <- renderUI({
    actionButton(inputId = "forecast_bttn",
                 icon = icon("tree-deciduous", lib = "glyphicon"),
                 style = "color: white;", 
                 label = "Make a Forecast!",
                 class = "btn btn-lg btn-primary"
    )
  })
  # 04.K remove_forecast_bttn ----
  output$remove_forecast_bttn <- renderUI({
    actionButton(inputId = "remove_forecast_bttn",
                 style = "color: white;", 
                 label = "Remove",
                 class = "btn btn-lg btn-danger"
    )
  })
  
  # 5 Forecast Panel ----
  
  output$forecast_panel <- renderUI({
    div(
      class = "jumbotron",
      div(
        class = "container bg-danger",
        #style = "background-color:#9999999c;",
        #style = "primary",
        h2("Forecast"),
        p("Select the Number of Days You'd Like to Forecast using ", code("R Shiny")),
        
        uiOutput("forecast"),
        uiOutput("forecast_bttn"), uiOutput("remove_forecast_bttn")
      )
    )
  })
  
  # 6 UI Sidebar Output ----
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
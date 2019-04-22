library(shiny)

shinyUI(fluidPage(
  titlePanel("Entropy Agglomeration - Işık Barış Fidaner"),
  fluidRow(
    textAreaInput("data","Enter feature allocation of elements:",width=600,height=250),
    actionLink("do","Example feature allocation"),
    br(),br(),
    actionButton("use","Use the feature allocation to draw a dendrogram")
  ),
  fluidRow(
    plotOutput("plot")
  )
)
  
  
)

# Alice Bob
# Alice David Eve
# Bob David
# Bob Carol Eve
# Carol David Eve
# Alice Eve

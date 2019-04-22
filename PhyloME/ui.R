library(shiny)

shinyUI(fluidPage(
  titlePanel("Phylogeny by Minimum Entropy - Işık Barış Fidaner"),
  fluidRow(
    textAreaInput("data","Enter aligned FASTA:",width=850,height=250),
    actionLink("do","Example aligned FASTA"),
    br(),br(),
    actionButton("use","Use aligned FASTA to draw a phylogenetic tree")
  ),
  fluidRow(
    plotOutput("plot")
  )
)
  
  
)

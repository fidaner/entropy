library(shiny)
library(shinycssloaders)

shinyUI(fluidPage(
  titlePanel("Phylogeny by Minimum Entropy - Işık Barış Fidaner"),
  fluidRow(
    textAreaInput("data","Enter aligned FASTA:",width=850,height=250),
    actionLink("do","Example aligned FASTA"),
    br(),br(),
    numericInput("bs","# Bootstrap",value=10,min=0,max=100,step=1),
    actionButton("use","Use aligned FASTA to draw a phylogenetic tree")
  ),
  fluidRow(
    textOutput("status"),
    withSpinner(plotOutput("plot")),
    textAreaInput("newick","Newick code:",width=850,height=100),
    downloadButton("downbtn","Download Newick")
  )
)
  
  
)

library(shiny)
library(shinycssloaders)

shinyUI(navbarPage(title = "PhyloME",
  tabPanel("Compute",
    fluidRow(
      textAreaInput("data","Enter aligned FASTA:",width=850,height=250),
      actionLink("do","Example aligned FASTA"),
      br(),br(),
      numericInput("bs","# Bootstrap",value=10,min=0,max=1000,step=1),
      textInput("email","For receiving e-mail:"),
      actionButton("use","Compute"),
      shinyjs::useShinyjs()
    ),
    fluidRow(
      textOutput("status"),
      withSpinner(plotOutput("plot")),
      textAreaInput("newick","Newick code:",width=850,height=100),
      downloadButton("downbtn","Download Newick")
    )
  ),
  tabPanel("Help",
           p("PhyloME (Phylogeny by Minimum Entropy) constructs a phylogenetic tree based on your protein/DNA/RNA sequences."),
           p("1. You must first do a multiple sequence alignment and prepare an aligned FASTA file."),
           p("2. Enter the contents of the aligned FASTA file to the PhyloME 'Compute' page."),
           p("(You can use the 'Example aligned FASTA' link to see an example)"),
           p("3. Enter the number of bootstrap instances. Enter zero if you don't want bootstrap computation."),
           p("4. Enter an email address to receive the Newick output (Recommended)."),
           p("5. Press the Compute button to start the computation."),
           p("6. The spinner animation will appear below the Compute button."),
           p("(If you have entered an e-mail address, you can close the page after pressing the Compute button. You will receive an e-mail with the Newick file when the computation finishes.)"),
           p("7. When the computation finishes, you can scroll down and click 'Download Newick'."),
           br(),br(),
           p("Işık Barış Fidaner & Athanasia Pavlopoulou")
           )
)
  
  
)

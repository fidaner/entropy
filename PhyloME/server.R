library(shiny)
library(rlist)

projection_entropy <- function( n, blocks, subset )
{
  pe = 0
  proj_indices = list()
  for(i in 1:length(blocks))
  {	proj_indices_i = c()
  for(j in 1:length(blocks[[i]]))
  {	if(blocks[[i]][j] %in% subset)
    proj_indices_i = c(proj_indices_i, j)
  }
  proj_indices[[i]] = proj_indices_i
  }
  
  for(i in 1:length(blocks))
  {	if(i<=length(proj_indices) && length(proj_indices[[i]]) != 0)
  {	p = 0
  for(j in proj_indices[[i]])
    p = p + 1
  p = p / ( length(subset) )
  if(p>0)
    pe = pe - p * log(p) / length(blocks)
  }
    
  }
  return(pe)
}



write_newick <- function( bifurcations, ind )
{
  if(ind>=1)
  {
    cat("(")
    pe = write_newick(bifurcations,bifurcations$merge[ind,1])
    cat(paste(":",bifurcations$height[ind]-pe,sep=""))
    cat(",")
    pe = write_newick(bifurcations,bifurcations$merge[ind,2])
    cat(paste(":",bifurcations$height[ind]-pe,sep=""))
    cat(")")
    #TODO: bootstrap
    return(bifurcations$height[ind])
  }
  else
  {
    cat(bifurcations$labels[-ind])
    return(0)
  }
}

write_newick2 <- function( str, bifurcations, ind )
{
  if(ind>=1)
  {
    str <- paste(str,"(",sep="")
    ret = write_newick2(str,bifurcations,bifurcations$merge[ind,1])
    str = ret$str
    pe = ret$pe
    str <- paste(str,paste(":",bifurcations$height[ind]-pe,sep=""),sep="")
    str <- paste(str,",",sep="")
    ret = write_newick2(str,bifurcations,bifurcations$merge[ind,2])
    str = ret$str
    pe = ret$pe
    str <- paste(str,paste(":",bifurcations$height[ind]-pe,sep=""),sep="")
    str <- paste(str,")",sep="")
    #TODO: bootstrap
    return(list(pe=bifurcations$height[ind],str=str))
  }
  else
  {
    str <- paste(str,bifurcations$labels[-ind],sep="")
    return(list(pe=0,str=str))
  }
}


agglomerate <- function(labels,blocks)
{
  n = length(labels)

  # begin with singleton clusters
  clusters = list()
  bfc_inds = c()
  for(i in 1:n)
  {	clusters[[i]] = c(i)
  bfc_inds = c(bfc_inds, -i)
  }
  bfc_count = n
  merge_entropies = list()
  
  # initialize the cache of minimum projection entropies
  min_pe_cache = list()
  min_pe_cache[[1]] = c(Inf,1,1)
  
  # compute the initial matrix and cache the cluster pairs with minimum projection entropy
  for(i in 1:n)
  {	entropies = c()
  if(i>1)
    for(j in 1:(i-1))
    {	pe = projection_entropy( n, blocks, c(clusters[[j]],clusters[[i]]) )
    entropies = c(entropies,pe)
    if(pe < min_pe_cache[[length(min_pe_cache)]][1])
      min_pe_cache[[length(min_pe_cache)+1]] = c(pe,j,i)
    }
  merge_entropies[[i]] = entropies
  }
  


  bifurcations = list()
  bifurcations_merge = matrix(ncol=2,nrow=0)
  bifurcations_height = c()
  
  # merge the best cluster pair and update the matrix
  # continue until only one cluster remains
  while(length(merge_entropies) > 1)
  {
    # print(bifurcations_merge)
    
    # record the bifurcation on the dendrogram
    bifurcations_merge <- rbind(bifurcations_merge, 0)
    bifurcations_height[nrow(bifurcations_merge)] =	min_pe_cache[[length(min_pe_cache)]][1]
    bifurcations_merge[nrow(bifurcations_merge),1] = bfc_inds[[min_pe_cache[[length(min_pe_cache)]][2]]]
    bifurcations_merge[nrow(bifurcations_merge),2] = bfc_inds[[min_pe_cache[[length(min_pe_cache)]][3]]]
    bfc_inds = bfc_inds[-min_pe_cache[[length(min_pe_cache)]][3]]
    bfc_count = bfc_count + 1
    bfc_inds[min_pe_cache[[length(min_pe_cache)]][2]] = bfc_count - n
    
    # print(merge_entropies)
    # print(min_pe_cache)
    # 
    # # log the event of merging
    # print(clusters[[min_pe_cache[[length(min_pe_cache)]][2]]])
    # print(clusters[[min_pe_cache[[length(min_pe_cache)]][3]]])
    # print(min_pe_cache[[length(min_pe_cache)]][1])
    
    # delete invalidated cache entries
    for(i in (length(min_pe_cache)-1):1)
      if(min_pe_cache[[i]][3] >= min_pe_cache[[length(min_pe_cache)]][2])
        min_pe_cache[[i]] <- NULL
    
    # remove the second cluster from the matrix
    merge_entropies[[min_pe_cache[[length(min_pe_cache)]][3]]] <- NULL
    if(min_pe_cache[[length(min_pe_cache)]][3] <= length(merge_entropies))
      for(i in min_pe_cache[[length(min_pe_cache)]][3]:length(merge_entropies))
      {
        merge_entropies[[i]] = merge_entropies[[i]][-min_pe_cache[[length(min_pe_cache)]][3]]
      }
    
    # merge the cluster pair to the first cluster
    clusters[[min_pe_cache[[length(min_pe_cache)]][2]]] = c(clusters[[min_pe_cache[[length(min_pe_cache)]][2]]],clusters[[min_pe_cache[[length(min_pe_cache)]][3]]])
    clusters[[min_pe_cache[[length(min_pe_cache)]][3]]] <- NULL
    
    # recompute the first cluster on the matrix
    j = min_pe_cache[[length(min_pe_cache)]][2]
    if(j+1 <= length(merge_entropies))
      for(i in (j+1):length(merge_entropies))
      {	pe = projection_entropy( n, blocks, c(clusters[[j]],clusters[[i]]) )
      merge_entropies[[i]][j] = pe
      }
    i = min_pe_cache[[length(min_pe_cache)]][2]
    if(i>1)
      for(j in 1:(i-1))
      {	pe = projection_entropy( n, blocks, c(clusters[[j]],clusters[[i]]) )
      merge_entropies[[i]][j] = pe
      }
    
    # delete the cache entry for the merged pair
    min_pe_cache[[length(min_pe_cache)]] <- NULL
    if(length(min_pe_cache) == 0)
      min_pe_cache[[1]] = c(Inf,1,1)
    
    # begin from the next entry
    # i0 = min_pe_cache[[length(min_pe_cache)]][3]
    
    # cache the cluster pairs with minimum projection entropy
    for(i in 1:length(merge_entropies))
      if(i>1)
        for(j in 1:(i-1))
          if(merge_entropies[[i]][j] < min_pe_cache[[length(min_pe_cache)]][1])
            min_pe_cache[[length(min_pe_cache)+1]] = c(merge_entropies[[i]][j],j,i)
    
  }
    

  
  bifurcations$merge = bifurcations_merge
  bifurcations$height = bifurcations_height
  bifurcations$order = 1:n
  bifurcations$labels = labels
  class(bifurcations) = 'hclust'

  return(bifurcations)
}

blockify <- function(seqs)
{
  blocks = list()
  for(i in 1:length(seqs))
  {
    pn = names(seqs)[i]
    for(j in 1:nchar(seqs[[pn]]))
      if(substring(seqs[[pn]],j,j)!='-')
      {
        feature = paste(j,substring(seqs[[pn]],j,j),sep=":")
        blocks[[feature]] = c(blocks[[feature]], i)
      }
  }
  return(blocks)
}



shinyServer( function(input,output,session) {
  
  observeEvent(input$do, {
    
    fileName <- 'aligned.txt'
    str <- readChar(fileName, file.info(fileName)$size)
    
    updateTextAreaInput(session, "data", value = str)
  })
  
  output$downbtn <- downloadHandler(
    filename = function() { "tree.newick" },
    content = function(file) { write(input$newick,file) }
  )
 
  output$plot <- renderPlot({
    
    input$use
    
    isolate({
      
      if(input$data!="")
      {
        # read FASTA
        
        data <- strsplit(input$data,"\n")[[1]]
        
        seqs = list()
        
        for(i in 1:length(data))
        {
          if(substring(data[i],1,1)=='>')
          {
            pn = substring(data[i],2)
            pn = gsub(",","_",pn)
            pn = gsub("\\(","[",pn)
            pn = gsub("\\)","]",pn)
            pn = gsub(";","_",pn)
            pn = gsub(" ","_",pn)
          }
          else
          {
            seqs[[pn]] = paste(seqs[[pn]], data[i], sep="")
          }
        }
        
        labels = names(seqs)

                

        num_bs = input$bs

        withProgress(message = 'Bootstrap', value = 0, {
          
          if(num_bs > 0)
          {
            clades = list()
            
            for(i in 1:num_bs)
            {
              bootstrap = seqs
              
              l = nchar(seqs[[labels[[1]]]])
              cols = sample(2:l,size=round(l/2))
              for(pn in labels)
              {
                newstr=''
                for(j in 1:l)
                  if(j %in% cols)
                    newstr = paste(newstr,substring(bootstrap[[pn]],j-1,j-1),sep="")
                  else
                    newstr = paste(newstr,substring(bootstrap[[pn]],j,j),sep="")
                bootstrap[[pn]] = newstr
              }
              
              blocks = blockify(bootstrap)
              
              bifurcations = agglomerate(labels, blocks)
              
              pop = list()
              for(j in 1:nrow(bifurcations$merge))
              {
                pop1 = c()
                for(k in 1:2)
                {
                  if(bifurcations$merge[j,k]<0)
                  {
                    pop1 = c(pop1,-bifurcations$merge[j,k])
                  }
                  else
                  {
                    pop1 = c(pop1,pop[[bifurcations$merge[j,k]]])
                  }
                }
                pop[[j]] = sort(pop1)
              }
              
              for(j in 1:length(pop))
              {
                clade = paste(pop[[j]],collapse=" ")
                
                if(is.null(clades[[clade]]))
                  clades[[clade]] = 0
                clades[[clade]] = clades[[clade]] + 1/num_bs
              }
              incProgress(1/num_bs)
            }
          }
        })        
        

                
        
        
        
        # read into blocks
        
        blocks = blockify(seqs)
        
        
        
        
        
        
        
        # start
        
        
        # data <- strsplit(strsplit(input$data,"\n")[[1]]," ")
        # 
        # print(data)
        # 
        # labels = c()
        # 
        # blocks = list()
        # for(i in 1:length(data))
        #   if(length(data[[i]])>0)
        #   { 
        #     block = c()
        #     for(j in 1:length(data[[i]]))
        #     {
        #       if(data[[i]][j] %in% labels)
        #       {
        #         k = which(data[[i]][j]==labels)
        #         block = c(block,k)
        #       }
        #       else
        #       {
        #         labels = c(labels,data[[i]][j])
        #         block = c(block,length(labels))
        #       }
        #     }
        #     if(length(block)>0)
        #       blocks = list.append(blocks,block)
        #   }  
        
        
        # DATA FROM THIS VIDEO:
        # https://www.youtube.com/watch?v=vNTaVc5q6sc
        
        # set the number of elements {Alice, Bob, Carol, David, Eve}
        n = length(labels)
        
        # labels = c('Alice','Bob','Carol','David','Eve')
        # 
        # # create the blocks from the elements.
        # block1 <- c(1,2)
        # block2 <- c(1,4,5)
        # block3 <- c(2,4)
        # block4 <- c(2,3,5)
        # block5 <- c(3,4,5)
        # block6 <- c(1,5)
        # 
        # blocks <- list(block1,block2,block3,block4,block5,block6)
        # 
        
        if(n>0)
        {
          bifurcations = agglomerate(labels, blocks)
          
          
          if(num_bs>0)
          {
            pop = list()
            for(j in 1:nrow(bifurcations$merge))
            {
              pop1 = c()
              for(k in 1:2)
              {
                if(bifurcations$merge[j,k]<0)
                {
                  pop1 = c(pop1,-bifurcations$merge[j,k])
                }
                else
                {
                  pop1 = c(pop1,pop[[bifurcations$merge[j,k]]])
                }
              }
              pop[[j]] = sort(pop1)
            }
            
            au = c()
            bp = c()
            for(j in 1:length(pop))
            {
              clade = paste(pop[[j]],collapse=" ")

              au[j] = NA
              bp[j] = clades[[clade]]
            }            
          }
          
          
          # SAVE NEWICK FILE
          # sink("tree.newick")
          # ind = nrow(bifurcations$merge)
          # pe = write_newick(bifurcations,ind)
          # cat(";")
          # sink()
          
          ind = nrow(bifurcations$merge)
          ret = write_newick2("",bifurcations,ind)
          ret$str = paste(ret$str,";",sep="")
          updateTextAreaInput(session, "newick", value = ret$str)
          

          # bifurcations = as.dendrogram(bifurcations)
          # plot(bifurcations,ylab="entropy")

          if(num_bs==0)
          {
            plot(bifurcations,ylab="entropy")
          }
          else
          {
            pv = list()
            pv$hclust = bifurcations
              pv$edges = data.frame(au=au,bp=bp)
            class(pv) = 'pvclust'
            plot(pv,ylab="entropy")
          }
          
          
        }
      }
      
    })

  }) 
  
})

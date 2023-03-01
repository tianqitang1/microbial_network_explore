#####################################################################
# The function for generating an interaction matrix
#
# Contact:
# Kazuhiro Takemoto (takemoto@bio.kyutech.ac.jp)
#####################################################################

generateM_specific_type <- function(nn,k_ave,type.network="random",type.interact="random",interact.str.max=0.5,mix.compt.ratio=0.5){
  # @param nn number of nodes
  # @param k_ave average degree (number of edges per node)
  # @param type.network network structure
  #               random: random networks
  #                   sf: scale-free networks
  #                   sw: small-world networks
  #                bipar: random bipartite networks
  # @param type.interact interaction type
  #               random: random
  #               mutual: mutalism (+/+ interaction)
  #                compt: competition (-/- interaction)
  #                   pp: predator-prey (+/- or -/+ interaction)
  #                  mix: mixture of mutualism and competition
  #                 mix2: mixture of competitive and predator-prey interactions
  # @param interact.str.max maximum interaction strength
  # @param mix.compt.ratio the ratio of competitive interactions to all intereactions (this parameter is only used for type.interact="mix" or ="mix2")

  # number of edges
  nl <- round(k_ave * nn / 2)
  if(type.network == "random"){
    g <- erdos.renyi.game(nn,nl,type="gnm")
  } else if(type.network == "sf"){
    g <- static.power.law.game(nn,nl,2.2,-1,loops = F,multiple = F,finite.size.correction = T)
  } else if(type.network == "sw") {
    g <- sample_smallworld(1, nn, round(k_ave / 2), 0.05, loops=F, multiple=F)
  } else if(type.network == "bipar") {
    g <- sample_bipartite(nn/2,nn/2,type="gnm",m=nl,directed=F)
  } else {
    stop("netwotk type is invalid")
  }

  # get adjacency matrix
  mtx_g <- as.matrix(get.adjacency(g))
  # get edge list
  edgelist <- get.edgelist(g)

  # generate an interaction matrix for the GLV model
  A <- matrix(0,nrow=nn,ncol = nn)

  if(type.interact == "random"){
    for(i in 1:nl){
      A[edgelist[i,1],edgelist[i,2]] <- runif(1,min=-interact.str.max,max=interact.str.max)
      A[edgelist[i,2],edgelist[i,1]] <- runif(1,min=-interact.str.max,max=interact.str.max)
    }
  } else if(type.interact == "mutual") {
    for(i in 1:nl){
      A[edgelist[i,1],edgelist[i,2]] <- runif(1,max=interact.str.max)
      A[edgelist[i,2],edgelist[i,1]] <- runif(1,max=interact.str.max)
    }
  } else if(type.interact == "compt"){
    for(i in 1:nl){
      A[edgelist[i,1],edgelist[i,2]] <- -runif(1,max=interact.str.max)
      A[edgelist[i,2],edgelist[i,1]] <- -runif(1,max=interact.str.max)
    }
  } else if(type.interact == "pp") {
    for(i in 1:nl){
      if(runif(1) < 0.5){
        A[edgelist[i,1],edgelist[i,2]] <- runif(1,max=interact.str.max)
        A[edgelist[i,2],edgelist[i,1]] <- -runif(1,max=interact.str.max)
      } else {
        A[edgelist[i,1],edgelist[i,2]] <- -runif(1,max=interact.str.max)
        A[edgelist[i,2],edgelist[i,1]] <- runif(1,max=interact.str.max)
      }
    }
  } else if(type.interact == "mix") {
    for(i in 1:nl){
      if(runif(1) < mix.compt.ratio){
        A[edgelist[i,1],edgelist[i,2]] <- -runif(1,max=interact.str.max)
        A[edgelist[i,2],edgelist[i,1]] <- -runif(1,max=interact.str.max)
      } else {
        A[edgelist[i,1],edgelist[i,2]] <- runif(1,max=interact.str.max)
        A[edgelist[i,2],edgelist[i,1]] <- runif(1,max=interact.str.max)
      }
    }
  } else if(type.interact == "mix2"){
    for(i in 1:nl){
      if(runif(1) < mix.compt.ratio){
        A[edgelist[i,1],edgelist[i,2]] <- -runif(1,max=interact.str.max)
        A[edgelist[i,2],edgelist[i,1]] <- -runif(1,max=interact.str.max)
      } else {
        if(runif(1) < 0.5){
          A[edgelist[i,1],edgelist[i,2]] <- runif(1,max=interact.str.max)
          A[edgelist[i,2],edgelist[i,1]] <- -runif(1,max=interact.str.max)
        } else {
          A[edgelist[i,1],edgelist[i,2]] <- -runif(1,max=interact.str.max)
          A[edgelist[i,2],edgelist[i,1]] <- runif(1,max=interact.str.max)
        }
      }
    }
  } else {
    stop("interaction type is invalid")
  }

  # diagonal elements
  diag(A) <- -1

  return(list(mtx_g,A))
}

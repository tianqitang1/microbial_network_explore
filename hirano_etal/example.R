#####################################################################
# Example R source code for generating datasets
# and for evaluating co-occurrence network performance.
#
# Contact:
# Kazuhiro Takemoto (takemoto@bio.kyutech.ac.jp)
#####################################################################

library(seqtime) # https://github.com/hallucigenia-sparsa/seqtime
library(SpiecEasi) # https://github.com/zdk123/SpiecEasi
library(igraph) # https://igraph.org/r/
library(ppcor) # https://cran.r-project.org/web/packages/ppcor/index.html
library(PRROC) # https://cran.r-project.org/package=PRROC
# load generateM_specific_type function
source("generateM_specific_type.R")

nn <- 50 # network size (n)
k_ave <- 2 # average degree (<k>)

## Genrate an interaction matrix (Mij)
obj <- generateM_specific_type(nn,k_ave,type.network="random",type.interact="random",interact.str.max=0.5,mix.compt.ratio=0.5)
# @param nn number of nodes
# @param k_ave average degree (number of edges per node)
# @param type.network network structure
#               random: random networks
#                   sf: scale-free networks
#                   ws: small-world networks
#                   bipar: random bipartite networks
# @param type.interact interaction type
#               random: random
#               mutual: mutalism (+/+ interaction)
#                compt: competition (-/- interaction)
#                   pp: predator-prey (+/- interaction)
#                  mix: mixture of mutualism and competition
#                 mix2: mixture of competitive and antagonistic interactions
# @param interact.str.max maximum interaction strength
# @param mix.compt.ratio the ratio of competitive interactions to all intereactions (this parameter is only used for type.interact="mix" or ="mix2")


# adjacency matrix of network (Aij)
network_real <- obj[[1]]
# interaction matrix (Mij) for the GLV model
M <- obj[[2]]

## plot population dynamics
y <- rpois(nn,lambda=100)
r <- runif(nn)
res <- glv(nn, M, r, y)
tsplot(10*res[,20:1000],time.given =T)

## Generate dataset on species abundance using the GLV model
data <- generateDataSet(300, M, count = nn*100, mode = 4)
# relative abundance
data_relative <- t(t(data) / apply(data,2,sum))
# plot correlation between species relative abundances
plot(as.data.frame(t(data_relative)))

## Inffering ecological associations (example)
# based on Pearson correlation
network_pred_pea <- abs(cor(t(data_relative)))
# based on Pearson partial correlation
network_pred_ppea <- abs(pcor(t(data_relative))$estimate)
# based on SparCC
# Note that in this study we used SparCC python module,
# not the follwing R wrapper function (see Methods section in the main text).
# (But, similar results are obtained)
network_pred_sparcc <- abs(sparcc(t(data))$Cor)
# based on SPIEC-EASI
network_pred_spiec <- spiec.easi(t(data),method='mb')
network_pred_spiec <- as.matrix(getOptMerge(network_pred_spiec))


## Evaluating evaluating co-occurrence network performance
# only use elements in lower triangular matrix
real <- network_real[lower.tri(network_real)] # Aij
pred_pea <- network_pred_pea[lower.tri(network_pred_pea)] # Pearson correlation
pred_ppea <- network_pred_ppea[lower.tri(network_pred_ppea)] # Pearson partial correlation
pred_sparcc <- network_pred_sparcc[lower.tri(network_pred_sparcc)] # SparCC
pred_spiec <- network_pred_spiec[lower.tri(network_pred_spiec)] #SPIEC-EASI


## Area under the Precision-Recall Curve (AUPR value)
pr.curve(pred_pea[real == 1],pred_pea[real == 0])$auc.integral # Pearson correlation
pr.curve(pred_ppea[real == 1],pred_ppea[real == 0])$auc.integral # Pearson partial correlation
pr.curve(pred_sparcc[real == 1],pred_sparcc[real == 0])$auc.integral # SparCC
pr.curve(pred_spiec[real == 1],pred_spiec[real == 0])$auc.integral #SPIEC-EASI

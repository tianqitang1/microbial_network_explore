library(metagenomeSeq)
library(edgeR)
library(DESeq2)
library(GUniFrac)
library(preprocessCore)

css.func <- function(tab) {
    tab_mrexperiment <- newMRexperiment(tab)
    tab_css <- cumNorm(tab_mrexperiment)
    tab_norm <- MRcounts(tab_css, norm = T)
    as.data.frame(tab_norm)
}

tmm.func <- function(tab) {
    tab <- as.matrix(tab)
    tab_dge <- DGEList(counts = tab)
    tab_tmm_dge <- calcNormFactors(tab_dge, method = "TMM")
    tab_norm <- cpm(tab_tmm_dge)
    as.data.frame(tab_norm)
}

tmmwsp.func <- function(tab) {
    tab <- as.matrix(tab)
    tab_dge <- DGEList(counts = tab)
    tab_tmm_dge <- calcNormFactors(tab_dge, method = "TMMwsp")
    tab_norm <- cpm(tab_tmm_dge)
    as.data.frame(tab_norm)
}

rle.func <- function(tab) {
    tab[tab == 0] <- 1
    metadata <- data.frame(class = factor(1:ncol(tab)))
    tab_dds <- DESeqDataSetFromMatrix(countData = tab, colData = metadata, design = ~class)
    tab_rle_dds <- estimateSizeFactors(tab_dds)
    tab_norm <- counts(tab_rle_dds, normalized = TRUE)
    as.data.frame(tab_norm)
}

gmpr.func <- function(tab){
      gmpr_size_factor <- GMPR(tab)
      tab_norm <- as.data.frame(t(t(tab)/gmpr_size_factor))
      as.data.frame(tab_norm)
    }

blom.func <- function(tab){
      tab <- as.matrix(tab)
      # a small noise term  might be added before data transformation to handle the ties
      noise <- matrix(rnorm(nrow(tab)*ncol(tab),mean=0,sd=10^-10),nrow=nrow(tab),ncol=ncol(tab))
      tab_noised <- tab+noise
      c <- 3/8
      tab_trans <- as.data.frame(t(apply(tab_noised,1,function(x) qnorm((rank(x)-c)/(ncol(tab_noised)-2*c+1)))))
      as.data.frame(tab_trans)
    }


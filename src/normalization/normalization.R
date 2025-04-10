########################################################################################################################
### normalization 
### normalize the training and testing separately: 
#   TSS, UQ, MED, CSS, CLR, CLR poscounts, logCPM, LOG, AST, 
### performed normalization on training, and then performed addon normalization of testing onto the training:
#   TMM, TMMwsp, TMM+, RLE, RLE poscounts, GMPR, STD, VST, NPN, rank, blom, QN, FSQN, BMC, combat, limma, conqur
### all the transformation methods were based on TSS normalized data
### all the batch correction methods were based on TSS+LOG normalized data (do not transformed back from log-space any more)
### the reference batch of batch correction methods should try train and test separately and then compare their performance
########################################################################################################################

#======================================================================================================================#
### function for merging two count table
merge.func <- function(table1,table2){
  table <- merge(table1,table2,by="row.names",all=T)
  rownames(table) <- table$Row.names
  table <- table[,-grep("Row.names",colnames(table))]
  table[is.na(table)] <- 0
  return(table)
}


#======================================================================================================================#
### function for normalize the data
norm.func <- function(p1,p2,norm_method){
  
  # TSS, for samples
  if(norm_method=="TSS"){
    norm_p1 <- as.data.frame(apply(p1,2,function(x) x/sum(x)))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) x/sum(x)))
    # let p2 have the same genes as p1 
    merged <- merge.func(norm_p1,norm_p2)
    final_p1 <- merged[rownames(norm_p1),colnames(norm_p1)]
    final_p2 <- merged[rownames(norm_p1),colnames(norm_p2)]
  }
  
  # UQ, for samples
  if(norm_method=="UQ"){
    norm_p1 <- as.data.frame(apply(p1,2,function(x) x/quantile(x[x>0])["75%"]))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) x/quantile(x[x>0])["75%"]))
    # let p2 have the same genes as p1 
    merged <- merge.func(norm_p1,norm_p2)
    final_p1 <- merged[rownames(norm_p1),colnames(norm_p1)]
    final_p2 <- merged[rownames(norm_p1),colnames(norm_p2)]
  }
  
  # MED, for samples
  if(norm_method=="MED"){
    norm_p1 <- as.data.frame(apply(p1,2,function(x) x/median(x[x>0])))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) x/median(x[x>0])))
    # let p2 have the same genes as p1 
    merged <- merge.func(norm_p1,norm_p2)
    final_p1 <- merged[rownames(norm_p1),colnames(norm_p1)]
    final_p2 <- merged[rownames(norm_p1),colnames(norm_p2)]
  }
  
  # CSS, for samples (the normalized quantile was selected for each sample separately)
  if(norm_method=="CSS"){
    require(metagenomeSeq)
    # function for CSS normalization
    css.func <- function(tab){
      tab_mrexperiment <- newMRexperiment(tab)
      tab_css <- cumNorm(tab_mrexperiment)
      tab_norm <- MRcounts(tab_css, norm=T)
      as.data.frame(tab_norm)
    }
    # css normalization
    norm_p1 <- css.func(p1)
    norm_p2 <- css.func(p2)
    # let p2 have the same genes as p1 
    merged <- merge.func(norm_p1,norm_p2)
    final_p1 <- merged[rownames(norm_p1),colnames(norm_p1)]
    final_p2 <- merged[rownames(norm_p1),colnames(norm_p2)]
  }
  
  # TMM, for samples, need to choose a reference
  # performed normalization on the training data, and then performed addon normalization of the test data onto the training data, 
  # to ensure that the normalization of the training data does not in any sense depend on the testing data.
  if(norm_method=="TMM"){
    require(edgeR)
    # function for TMM normalization
    tmm.func <- function(tab){
      tab <- as.matrix(tab)
      tab_dge <- DGEList(counts=tab)
      tab_tmm_dge <- calcNormFactors(tab_dge, method="TMM")
      tab_norm <- cpm(tab_tmm_dge)
      as.data.frame(tab_norm)
    }
    # TMM normalization,  performed transformation on the p1, and then performed addon transformation of p2 onto p1, 
    # to ensure that the transformation of the p1 does not in any sense depend on the p2.
    norm_p1 <- tmm.func(p1)
    norm_p2 <- tmm.func(merge.func(p1,p2))[rownames(p1),colnames(p2)]
    # let p2 have the same genes as p1 
    final_p1 <- norm_p1
    final_p2 <- norm_p2
  }
  
  # TMM+, for samples, need to choose a reference
  # performed normalization on the training data, and then performed addon normalization of the test data onto the training data, 
  # to ensure that the normalization of the training data does not in any sense depend on the testing data.
  if(norm_method=="TMM+"){
    require(edgeR)
    # function for TMM+ normalization
    tmm.func <- function(tab){
      tab[tab==0] <- 1
      tab <- as.matrix(tab)
      tab_dge <- DGEList(counts=tab)
      tab_tmm_dge <- calcNormFactors(tab_dge, method="TMM")
      tab_norm <- cpm(tab_tmm_dge)
      as.data.frame(tab_norm)
    }
    # TMM+ normalization
    norm_p1 <- tmm.func(p1)
    norm_p2 <- tmm.func(merge.func(p1,p2))[rownames(p1),colnames(p2)]
    # let p2 have the same genes as p1 
    final_p1 <- norm_p1
    final_p2 <- norm_p2
  }
  
  # TMMwsp, TMM with singleton pairing, an alternative for highly sparse data
  # need to choose a reference, the column with largest sum of square-root counts is used as the reference library
  # performed normalization on the training data, and then performed addon normalization of the test data onto the training data, 
  # to ensure that the normalization of the training data does not in any sense depend on the testing data.
  if(norm_method=="TMMwsp"){
    require(edgeR)
    # function for TMMwsp normalization
    tmmwsp.func <- function(tab){
      tab <- as.matrix(tab)
      tab_dge <- DGEList(counts=tab)
      tab_tmm_dge <- calcNormFactors(tab_dge, method="TMMwsp")
      tab_norm <- cpm(tab_tmm_dge)
      as.data.frame(tab_norm)
    }
    # TMMwsp normalization
    norm_p1 <- tmmwsp.func(p1)
    norm_p2 <- tmmwsp.func(merge.func(p1,p2))[rownames(p1),colnames(p2)]
    # let p2 have the same genes as p1 
    final_p1 <- norm_p1
    final_p2 <- norm_p2
  }
  
  # RLE, for samples
  # Each column is divided by the geometric means of the rows. The median of these ratios is used as the size factor for this column.
  # add a pseudo count 1 to avoid a geometric mean of zero
  # performed normalization on the training data, and then performed addon normalization of the test data onto the training data, 
  # to ensure that the normalization of the training data does not in any sense depend on the testing data.
  if(norm_method=="RLE"){
    require(DESeq2)
    # function for RLE normalization
    rle.func <- function(tab){
      tab[tab==0] <- 1
      metadata <- data.frame(class=factor(1:ncol(tab)))
      tab_dds <- DESeqDataSetFromMatrix(countData=tab,colData=metadata,design=~class) 
      tab_rle_dds <- estimateSizeFactors(tab_dds)
      tab_norm <- counts(tab_rle_dds, normalized=TRUE)
      as.data.frame(tab_norm)
    }
    # RLE normalization
    norm_p1 <- rle.func(p1)
    norm_p2 <- rle.func(merge.func(p1,p2))[rownames(p1),colnames(p2)]
    # let p2 have the same genes as p1 
    final_p1 <- norm_p1
    final_p2 <- norm_p2
  }
  
  # RLE_poscounts, for samples
  # Each column is divided by the geometric means of the rows. The median of these ratios is used as the size factor for this column.
  # use the none zero counts to calculate the geometric mean
  if(norm_method=="RLE_poscounts"){
    require(DESeq2)
    # function for RLE with poscounts estimator normalization
    rle.poscounts.func <- function(tab){
      metadata <- data.frame(class=factor(1:ncol(tab)))
      tab_dds <- DESeqDataSetFromMatrix(countData=tab,colData=metadata,design=~class) 
      tab_rle_dds <- estimateSizeFactors(tab_dds,type="poscounts")
      tab_norm <- counts(tab_rle_dds, normalized=TRUE)
      as.data.frame(tab_norm)
    }
    # RLE normalization
    norm_p1 <- rle.poscounts.func(p1)
    norm_p2 <- rle.poscounts.func(merge.func(p1,p2))[rownames(p1),colnames(p2)]
    # let p2 have the same genes as p1 
    final_p1 <- norm_p1
    final_p2 <- norm_p2
  }
  
  # GMPR, for samples
  # switching the two steps in RLE(DESeq2) normalization
  if(norm_method=="GMPR"){
    require(GUniFrac)
    # function for GMPR normalization
    gmpr.func <- function(tab){
      gmpr_size_factor <- GMPR(tab)
      tab_norm <- as.data.frame(t(t(tab)/gmpr_size_factor))
      as.data.frame(tab_norm)
    }
    # GMPR normalization, performed transformation on the p1, and then performed addon transformation of p2 onto p1, 
    # to ensure that the transformation of the p1 does not in any sense depend on the p2.
    norm_p1 <- gmpr.func(p1)
    norm_p2 <- gmpr.func(merge.func(p1,p2))[rownames(p1),colnames(p2)]
    # let p2 have the same genes as p1 
    final_p1 <- norm_p1
    final_p2 <- norm_p2
  }
  
  # CLR, for samples
  # transform to relative abundance first, add a pseudo count 0.65*minimum to zero values
  if(norm_method=="CLR"){
    require(compositions)
    # replaced all the 0 abundances with 0.65 times minimum non-zero abundance
    p1[p1==0] <- min(p1[p1!=0])*0.65
    p2[p2==0] <- min(p2[p2!=0])*0.65
    # clr transformation
    norm_p1 <- as.data.frame(apply(p1,2,function(x) clr(x)))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) clr(x)))
    # let p2 have the same genes as p1 
    merged <- merge.func(norm_p1,norm_p2)
    final_p1 <- merged[rownames(norm_p1),colnames(norm_p1)]
    final_p2 <- merged[rownames(norm_p1),colnames(norm_p2)]
  }
  
  # CLR_poscounts, for samples
  # transform to relative abundance first, use the positive counts only
  if(norm_method=="CLR_poscounts"){
    require(compositions)
    # clr transformation
    norm_p1 <- as.data.frame(apply(p1,2,function(x) clr(x)))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) clr(x)))
    # let p2 have the same genes as p1 
    merged <- merge.func(norm_p1,norm_p2)
    final_p1 <- merged[rownames(norm_p1),colnames(norm_p1)]
    final_p2 <- merged[rownames(norm_p1),colnames(norm_p2)]
  }
  
  # logcpm, for samples
  # Compute counts per million (CPM)
  if(norm_method=="logcpm"){
    require(edgeR)
    # replaced all the 0 abundances with 1 (need integers as input)
    p1[p1==0] <- 1
    p2[p2==0] <- 1
    # logcpm normalization
    norm_p1 <- as.data.frame(cpm(p1,log=TRUE))
    norm_p2 <- as.data.frame(cpm(p2,log=TRUE))
    # let p2 have the same genes as p1 
    merged <- merge.func(norm_p1,norm_p2)
    final_p1 <- merged[rownames(norm_p1),colnames(norm_p1)]
    final_p2 <- merged[rownames(norm_p1),colnames(norm_p2)]
  }
  
  # LOG, for genes
  # transform to relative abundance first, add a pseudo count 0.65*minimum to zero values
  if(norm_method=="LOG"){
    # relative abundance
    norm_p1 <- as.data.frame(apply(p1,2,function(x) x/sum(x)))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) x/sum(x)))
    # replaced all the 0 abundances with 0.65 times minimum non-zero abundance
    norm_p1[norm_p1==0] <- min(norm_p1[norm_p1!=0])*0.65
    norm_p2[norm_p2==0] <- min(norm_p2[norm_p2!=0])*0.65
    # log transformation
    trans_p1 <- log(norm_p1)
    trans_p2 <- log(norm_p2)
    # let p2 have the same genes as p1 
    merged <- merge.func(trans_p1,trans_p2)
    final_p1 <- merged[rownames(trans_p1),colnames(trans_p1)]
    final_p2 <- merged[rownames(trans_p1),colnames(trans_p2)]
  }
  
  # AST, for genes
  # transform to relative abundance first, then do the arcsine square root transformation
  if(norm_method=="AST"){
    # relative abundance
    norm_p1 <- as.data.frame(apply(p1,2,function(x) x/sum(x)))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) x/sum(x)))
    # AST transformation
    trans_p1 <- asin(sqrt(norm_p1))
    trans_p2 <- asin(sqrt(norm_p2))
    # let p2 have the same genes as p1 
    merged <- merge.func(trans_p1,trans_p2)
    final_p1 <- merged[rownames(trans_p1),colnames(trans_p1)]
    final_p2 <- merged[rownames(trans_p1),colnames(trans_p2)]
  }
  
  # STD, for genes
  # transform to relative abundance first, substract the mean and devided by the standard deviation
  if(norm_method=="STD"){
    # relative abundance
    norm_p1 <- as.data.frame(apply(p1,2,function(x) x/sum(x)))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) x/sum(x)))
    # function for standardization
    std.func <- function(tab){
      tab_trans <- as.data.frame(t(apply(tab,1,function(x) (x-mean(x))/sd(x))))
      tab_trans
    }
    # performed transformation on the p1, and then performed addon transformation of p2 onto the training data, 
    # to ensure that the transformation of the p1 does not in any sense depend on the p2.
    trans_p1 <- std.func(norm_p1)
    trans_p2 <- std.func(merge.func(norm_p1,norm_p2))[rownames(norm_p1),colnames(norm_p2)]
    # let p2 have the same genes as p1 
    final_p1 <- trans_p1
    final_p2 <- trans_p2
  }
  
  # rank, for genes
  # transform to relative abundance first, then do the rank transformation for each genes
  if(norm_method=="rank"){
    # relative abundance
    norm_p1 <- as.data.frame(apply(p1,2,function(x) x/sum(x)))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) x/sum(x)))
    # function for rank transformation
    rank.func <- function(tab){
      tab <- as.matrix(tab)
      # a small noise term  might be added before data transformation to handle the ties
      noise <- matrix(rnorm(nrow(tab)*ncol(tab),mean=0,sd=10^-10),nrow=nrow(tab),ncol=ncol(tab))
      tab_noised <- tab+noise
      tab_trans <- as.data.frame(t(apply(tab_noised,1,rank)))
      tab_trans
    }
    # performed transformation on the p1, and then performed addon transformation of p2 onto the training data, 
    # to ensure that the transformation of the p1 does not in any sense depend on the p2.
    trans_p1 <- rank.func(norm_p1)
    trans_p2 <- rank.func(merge.func(norm_p1,norm_p2))[rownames(norm_p1),colnames(norm_p2)]
    # let p2 have the same genes as p1 
    final_p1 <- trans_p1
    final_p2 <- trans_p2
  }
  
  # blom, for genes
  # transform to relative abundance first, then do the blom transformation (rank transformation to normality) for each genes
  if(norm_method=="blom"){
    # relative abundance
    norm_p1 <- as.data.frame(apply(p1,2,function(x) x/sum(x)))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) x/sum(x)))
    # function for blom transformation
    blom.func <- function(tab){
      tab <- as.matrix(tab)
      # a small noise term  might be added before data transformation to handle the ties
      noise <- matrix(rnorm(nrow(tab)*ncol(tab),mean=0,sd=10^-10),nrow=nrow(tab),ncol=ncol(tab))
      tab_noised <- tab+noise
      c <- 3/8
      tab_trans <- as.data.frame(t(apply(tab_noised,1,function(x) qnorm((rank(x)-c)/(ncol(tab_noised)-2*c+1)))))
      as.data.frame(tab_trans)
    }
    # performed transformation on the p1, and then performed addon transformation of p2 onto the training data, 
    # to ensure that the transformation of the p1 does not in any sense depend on the p2.
    trans_p1 <- blom.func(norm_p1)
    trans_p2 <- blom.func(merge.func(norm_p1,norm_p2))[rownames(norm_p1),colnames(norm_p2)]
    # let p2 have the same genes as p1 
    final_p1 <- trans_p1
    final_p2 <- trans_p2
  }
  
  # VST, for genes
  # DESeq2 package need integer as input, add pseudo count 1 to zero values
  if(norm_method=="VST"){
    require(DESeq2)
    # replaced all the 0 abundances with 1
    p1[p1==0] <- 1
    p2[p2==0] <- 1
    # VST transformation
    # performed transformation on the p1, and then performed addon transformation of p2 onto the training data, 
    # to ensure that the transformation of the p1 does not in any sense depend on the p2.
    trans_p1 <- as.data.frame(varianceStabilizingTransformation(as.matrix(p1)))
    trans_p2 <- as.data.frame(varianceStabilizingTransformation(as.matrix(merge.func(p1,p2))))[rownames(p1),colnames(p2)]
    # let p2 have the same genes as p1 
    final_p1 <- trans_p1
    final_p2 <- trans_p2
  }
  
  # NPN, for genes, based on relative abundance
  if(norm_method=="NPN"){
    require(huge)
    # relative abundance
    norm_p1 <- as.data.frame(apply(p1,2,function(x) x/sum(x)))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) x/sum(x)))
    # Nonparanormal(npn) transformation
    # performed transformation on the p1, and then performed addon transformation of p2 onto the training data, 
    # to ensure that the transformation of the p1 does not in any sense depend on the p2.
    trans_p1 <- as.data.frame(t(huge.npn(t(norm_p1),npn.func="truncation")))
    trans_p2 <- as.data.frame(t(huge.npn(t(merge.func(norm_p1,norm_p2)),npn.func="truncation")))[rownames(norm_p1),colnames(norm_p2)]  
    # let p2 have the same genes as p1 
    final_p1 <- trans_p1
    final_p2 <- trans_p2
  }
  
  # QN_trn, for samples, batch correction, based on log transformed relative abundance, using traning as reference 
  if(norm_method=="QN_trn"){
    require(preprocessCore)
    # relative abundance
    norm_p1 <- as.data.frame(apply(p1,2,function(x) x/sum(x)))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) x/sum(x)))
    # replaced all the 0 abundances with 0.65 times minimum non-zero abundance
    norm_p1[norm_p1==0] <- min(norm_p1[norm_p1!=0])*0.65
    norm_p2[norm_p2==0] <- min(norm_p2[norm_p2!=0])*0.65
    # log transformation
    trans_p1 <- log(norm_p1)
    trans_p2 <- log(norm_p2)
    # reference quantiles, testing set
    ref_quantiles <- normalize.quantiles.determine.target(x=as.matrix(trans_p1))
    # quantile normalize norm_p1 and norm_p2, using the ref_quantiles as reference
    correct_p1 <- normalize.quantiles(as.matrix(trans_p1))
    dimnames(correct_p1) <- dimnames(p1)
    merged <- merge.func(trans_p1,trans_p2)
    norm_merged <- normalize.quantiles.use.target(as.matrix(merged),target=ref_quantiles)
    dimnames(norm_merged) <- dimnames(merged)
    correct_p2 <- as.data.frame(norm_merged[rownames(p1),colnames(p2)])
    # let p2 have the same genes as p1 
    final_p1 <- correct_p1
    final_p2 <- correct_p2
  }
  
  
  # QN_tst, for samples, batch correction, based on log transformed relative abundance, using testing as reference 
  if(norm_method=="QN_tst"){
    require(preprocessCore)
    # relative abundance
    norm_p1 <- as.data.frame(apply(p1,2,function(x) x/sum(x)))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) x/sum(x)))
    # replaced all the 0 abundances with 0.65 times minimum non-zero abundance
    norm_p1[norm_p1==0] <- min(norm_p1[norm_p1!=0])*0.65
    norm_p2[norm_p2==0] <- min(norm_p2[norm_p2!=0])*0.65
    # log transformation
    trans_p1 <- log(norm_p1)
    trans_p2 <- log(norm_p2)
    # reference quantiles, testing set
    ref_quantiles <- normalize.quantiles.determine.target(x=as.matrix(trans_p2))
    # quantile normalize norm_p1 and norm_p2, using the ref_quantiles as reference
    correct_p1 <- normalize.quantiles(as.matrix(trans_p1))
    dimnames(correct_p1) <- dimnames(p1)
    merged <- merge.func(trans_p1,trans_p2)
    norm_merged <- normalize.quantiles.use.target(as.matrix(merged),target=ref_quantiles)
    dimnames(norm_merged) <- dimnames(merged)
    correct_p2 <- as.data.frame(norm_merged[rownames(p1),colnames(p2)])
    # let p2 have the same genes as p1 
    final_p1 <- correct_p1
    final_p2 <- correct_p2
  }
  
  # FSQN, for genes, batch correction, based on log transformed relative abundance
  if(norm_method=="FSQN"){
    require(FSQN)
    # relative abundance
    norm_p1 <- as.data.frame(apply(p1,2,function(x) x/sum(x)))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) x/sum(x)))
    # replaced all the 0 abundances with 0.65 times minimum non-zero abundance
    norm_p1[norm_p1==0] <- min(norm_p1[norm_p1!=0])*0.65
    norm_p2[norm_p2==0] <- min(norm_p2[norm_p2!=0])*0.65
    # log transformation
    trans_p1 <- log(norm_p1)
    trans_p2 <- log(norm_p2)
    # FSQN
    correct_p1 <- trans_p1
    correct_p2 <- merge.func(trans_p1,trans_p2)[rownames(trans_p1),colnames(trans_p2)]
    correct_p2 <- t(quantileNormalizeByFeature(matrix_to_normalize=as.matrix(t(correct_p2)),
                                               target_distribution_matrix=as.matrix(t(correct_p1))))
    # let p2 have the same genes as p1 
    final_p1 <- correct_p1
    final_p2 <- correct_p2
  }
  
  # bmc, for genes, batch correction, based on log transformed relative abundance
  if(norm_method=="BMC"){
    require(pamr)
    # relative abundance
    norm_p1 <- as.data.frame(apply(p1,2,function(x) x/sum(x)))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) x/sum(x)))
    # replaced all the 0 abundances with 0.65 times minimum non-zero abundance
    norm_p1[norm_p1==0] <- min(norm_p1[norm_p1!=0])*0.65
    norm_p2[norm_p2==0] <- min(norm_p2[norm_p2!=0])*0.65
    # log transformation
    trans_p1 <- log(norm_p1)
    trans_p2 <- log(norm_p2)
    # bmc correction
    merged <- merge.func(trans_p1,trans_p2)
    batch_factor <- factor(c(rep(1,ncol(trans_p1)),rep(2,ncol(trans_p2))))
    correct_merged <- as.data.frame(pamr.batchadjust(list(x=as.matrix(merged),batchlabels=batch_factor))$x)
    # let p2 have the same genes as p1 
    final_p1 <- correct_merged[rownames(trans_p1),colnames(trans_p1)]
    final_p2 <- correct_merged[rownames(trans_p1),colnames(trans_p2)]
  }
  
  # limma, batch correction
  # relative abundances were log transformed prior to combat, corrected data were then transformed back from log space (ref percentile normalization)
  if(norm_method=="limma"){
    require(limma)
    # relative abundance
    norm_p1 <- as.data.frame(apply(p1,2,function(x) x/sum(x)))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) x/sum(x)))
    # replaced all the 0 abundances with 0.65 times minimum non-zero abundance
    norm_p1[norm_p1==0] <- min(norm_p1[norm_p1!=0])*0.65
    norm_p2[norm_p2==0] <- min(norm_p2[norm_p2!=0])*0.65
    # log transformation
    trans_p1 <- log(norm_p1)
    trans_p2 <- log(norm_p2)
    #limma
    merged <- merge.func(trans_p1,trans_p2)
    batch_factor <- factor(c(rep(1,ncol(trans_p1)),rep(2,ncol(trans_p2))))
    correct_merged <- as.data.frame(removeBatchEffect(merged, batch=batch_factor))
    # let p2 have the same genes as p1 
    final_p1 <- correct_merged[rownames(trans_p1),colnames(trans_p1)]
    final_p2 <- correct_merged[rownames(trans_p1),colnames(trans_p2)]
  }
  
  # combat_trn, batch correction, based on log-transformed relative abundance, with training as reference batch
  if(norm_method=="combat_trn"){
    require(sva)
    # relative abundance
    norm_p1 <- as.data.frame(apply(p1,2,function(x) x/sum(x)))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) x/sum(x)))
    # replaced all the 0 abundances with 0.65 times minimum non-zero abundance
    norm_p1[norm_p1==0] <- min(norm_p1[norm_p1!=0])*0.65
    norm_p2[norm_p2==0] <- min(norm_p2[norm_p2!=0])*0.65
    # log transformation
    trans_p1 <- log(norm_p1)
    trans_p2 <- log(norm_p2)
    # combat
    merged <- merge.func(trans_p1,trans_p2)
    batch_factor <- factor(c(rep(1,ncol(trans_p1)),rep(2,ncol(trans_p2))))
    correct_merged <- as.data.frame(ComBat(merged, batch=batch_factor,ref.batch=1))
    # let p2 have the same genes as p1 
    final_p1 <- correct_merged[rownames(trans_p1),colnames(trans_p1)]
    final_p2 <- correct_merged[rownames(trans_p1),colnames(trans_p2)]
  }
  
  # combat_tst, batch correction, based on log-transformed relative abundance, with testing as reference batch
  if(norm_method=="combat_tst"){
    require(sva)
    # relative abundance
    norm_p1 <- as.data.frame(apply(p1,2,function(x) x/sum(x)))
    norm_p2 <- as.data.frame(apply(p2,2,function(x) x/sum(x)))
    # replaced all the 0 abundances with 0.65 times minimum non-zero abundance
    norm_p1[norm_p1==0] <- min(norm_p1[norm_p1!=0])*0.65
    norm_p2[norm_p2==0] <- min(norm_p2[norm_p2!=0])*0.65
    # log transformation
    trans_p1 <- log(norm_p1)
    trans_p2 <- log(norm_p2)
    # combat
    merged <- merge.func(trans_p1,trans_p2)
    batch_factor <- factor(c(rep(1,ncol(trans_p1)),rep(2,ncol(trans_p2))))
    correct_merged <- as.data.frame(ComBat(merged, batch=batch_factor,ref.batch=2))
    # let p2 have the same genes as p1 
    final_p1 <- correct_merged[rownames(trans_p1),colnames(trans_p1)]
    final_p2 <- correct_merged[rownames(trans_p1),colnames(trans_p2)]
  }
  
  # conqur_trn, batch correction, using training data as reference
  # directly worked on the taxa read count table
  if(norm_method=="conqur_trn"){
    require(ConQuR)
    require(foreach)
    merged <- merge.func(p1,p2)
    batch_factor <- factor(c(rep(1,ncol(p1)),rep(2,ncol(p2))))
    covariates <- data.frame(covariate=rep(1,ncol(merged)))   # no additional information could be used for each dataset
    correct_merged <- ConQuR(tax_tab=as.data.frame(t(merged)),batchid=batch_factor,batch_ref="1",covariates=covariates,simple_match=T)
    correct_merged <- as.data.frame(t(correct_merged))
    ## normalize the corrected count data
    #norm_merged <- as.data.frame(apply(correct_merged,2,function(x) x/sum(x)))
    # let p2 have the same genes as p1 
    final_p1 <- correct_merged[rownames(p1),colnames(p1)]
    final_p2 <- correct_merged[rownames(p1),colnames(p2)]
  }
  
  # conqur_tst, batch correction,using testing data as reference batch
  # directly worked on the taxa read count table
  if(norm_method=="conqur_tst"){
    require(ConQuR)
    require(foreach)
    merged <- merge.func(p1,p2)
    batch_factor <- factor(c(rep(1,ncol(p1)),rep(2,ncol(p2))))
    covariates <- data.frame(covariate=rep(1,ncol(merged)))   # no additional information could be used for each dataset
    correct_merged <- ConQuR(tax_tab=as.data.frame(t(merged)),batchid=batch_factor,batch_ref="2",covariates=covariates,simple_match=T)
    correct_merged <- as.data.frame(t(correct_merged))
    ## normalize the corrected count data
    #norm_merged <- as.data.frame(apply(correct_merged,2,function(x) x/sum(x)))
    # let p2 have the same genes as p1 
    final_p1 <- correct_merged[rownames(p1),colnames(p1)]
    final_p2 <- correct_merged[rownames(p1),colnames(p2)]
  }
  
  # return
  list(final_p1,final_p2)
}









# Normalization methods

There are many different methods for normalizing the taxonomic abundance data.
Suppose the abundance data is $X_{ij}$, where $i$ is the sample index and $j$ is the species index.

The normalization methods can be divided into two categories: methods that require training and methods that do not require training.


## Total sum scaling (TSS)

The total sum scaling (TSS) method normalizes the data by dividing each sample by the total sum of the sample.

$$
X_{ij} = \frac{X_{ij}}{\sum_{j=1}^{N} X_{ij}}
$$

## Upper quartile scaling (UQS)

The upper quartile scaling (UQS) method normalizes the data by dividing each sample by the upper quartile of the sample.
Only positive values are considered.

$$
X_{ij} = \frac{X_{ij}}{Q_{0.75}(X_{i})}
$$

## Median scaling (MED)

The Median scaling (MED) method normalizes the data by dividing each sample by the median of the sample.
Only positive values are considered.

$$
X_{ij} = \frac{X_{ij}}{Q_{0.5}(X_{i})}
$$

## Cumulative sum scaling (CSS)

Cumulative Sum Scaling (CSS) is a median-like quantile normalization which corrects differences in sampling depth (library size). While standard relative abundance (fraction/percentage) normalization re-scales all samples to the same total sum (100%), CSS keeps a variation in total counts between samples. CSS re-scales the samples based on a subset (quartile) of lower abundant taxa (relatively constant and independent), thereby excluding the impact of (study dominating) high abundant taxa.

This one seems to work well.

## TMM normalization

## TMM+

## TMMwsp

TMM with singleton pairing

## RLE

## RLE poscounts

## GMPR

## CLR

Centered log-ratio transformation (CLR) is a method for transforming compositional data into real values.

$$
X_{ij} = \log \frac{X_{ij}}{G_{i}}
$$

where $G_{i}$ is the geometric mean of the sample $i$.

## CLR_poscounts

CLR with positive counts only

## logcpm

## LOG

## Arcsine square root transformation (AST)

## STD

## Rank transformation

## BLOM transformation

## NPN

## QN

## FSQN

## BMC

## LIMMA

## ComBat

## ConQuR
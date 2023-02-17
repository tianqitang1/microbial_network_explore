library(reticulate)
library(tidyverse)
library(yaml)
library(NetCoMi)
library(PRROC)
library(glue)
np <- import("numpy")

set.seed(42)

data_dir <- 'data/simulated/n20_k4_random_random_1.0'

z <- np$load(glue("{data_dir}/z.npy"))
x <- np$load(glue("{data_dir}/x.npy"))
y <- np$load(glue("{data_dir}/y.npy"))
adj <- np$load(glue("{data_dir}/adj.npy"))
M <- np$load(glue("{data_dir}/M.npy"))

z <- t(z)
x <- t(x)
y <- t(y)

# Evaluate Pearson
pred_pea <- abs(cor(y))
pred_pea <- pred_pea[lower.tri(pred_pea)]
print(pr.curve(pred_pea[target == 1], pred_pea[target == 0])$auc.integral)

# Evaluate CCLasso
pred <- NetCoMi::cclasso(t(y), counts = TRUE, pseudo = 0.01)
pred <- pred$cor.w
pred <- pred[lower.tri(pred)]
print(pr.curve(pred[target == 1], pred[target == 0])$auc.integral)
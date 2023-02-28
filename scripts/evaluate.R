library(reticulate)
library(tidyverse)
library(yaml)
library(NetCoMi)
library(PRROC)
library(glue)
np <- import("numpy")

set.seed(42)

data_dir <- 'data/simulated/n20_k4_random_random_1.0'

# config <- yaml::read_yaml("scripts/config.yaml")
# scale_simulation <- config$simulation$scale_simulation
scale_simulation <- 100

z <- np$load(glue("{data_dir}/z.npy"))
x <- np$load(glue("{data_dir}/x.npy"))
y <- np$load(glue("{data_dir}/y.npy"))
adj <- np$load(glue("{data_dir}/adj.npy"))
M <- np$load(glue("{data_dir}/M.npy"))

z <- t(z)
x <- t(x)
y <- t(y)
y <- y * scale_simulation

target <- c(adj[lower.tri(adj)], adj[upper.tri(adj)])

# Generate baseline matrix filled with random number between 0 and 1
baseline <- matrix(runif(n_vertices * n_vertices), nrow = n_vertices, ncol = n_vertices)
# Take upper and lower triangle of the baseline matrix
baseline <- c(baseline[lower.tri(baseline)], baseline[upper.tri(baseline)])
baseline_prauc <- pr.curve(baseline[target == 1], baseline[target == 0])$auc.integral
cat("Baseline:\t", baseline_prauc, "\n")

# Evaluate Pearson
pred_pea <- abs(cor(y))
pred_pea <- c(pred_pea[lower.tri(pred_pea)], pred_pea[upper.tri(pred_pea)])
cat("Pearson:\t", pr.curve(pred_pea[target == 1], pred_pea[target == 0])$auc.integral, "\n")

# Evaluate CCLasso
pred <- NetCoMi::cclasso(t(y), counts = TRUE, pseudo = 0.01)
pred <- abs(pred$cor.w)
pred <- c(pred[lower.tri(pred)], pred[upper.tri(pred)])
cat("CCLasso:\t", pr.curve(pred[target == 1], pred[target == 0])$auc.integral, "\n")
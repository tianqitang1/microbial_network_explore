library(reticulate)
library(tidyverse)
library(yaml)
library(NetCoMi)
library(PRROC)
library(glue)

set.seed(42)

config <- yaml::read_yaml("scripts/config.yaml")
n_vertices <- config$network$n_vertices
avg_degree <- config$network$avg_degree
time_points <- config$simulation$time_points
time_step <- config$simulation$time_step
downsample <- config$simulation$downsample
noise_var <- config$simulation$noise_var
scale_simulation <- config$simulation$scale_simulation

data_dir <- config$paths$data_dir

# Load the Python module
src <- reticulate::import("src")
simulate_glv <- src$utils$simulate_glv

# Simulate the time series data using gLV model
simulate_glv(num_taxa = n_vertices, avg_degree = avg_degree, interaction_type='mutualism') %>%
  {
    z <<- .[[1]]
    x <<- .[[2]]
    y <<- .[[3]]
    adj <<- .[[4]]
    M <<- .[[5]]
  }


target <- adj[lower.tri(adj)]

# Downsample the original raw data
z <- t(z)
# z <- z[, seq.int(1, time_points, downsample)]

y <- t(y)
y <- y * scale_factor

# Evaluate Pearson
pred_pea <- abs(cor(y))
pred_pea <- pred_pea[lower.tri(pred_pea)]
print(pr.curve(pred_pea[target == 1], pred_pea[target == 0])$auc.integral)

pred <- NetCoMi::cclasso(t(y), counts = TRUE, pseudo = 0.01)
pred <- abs(pred$cor.w)
pred <- pred[lower.tri(pred)]
print(pr.curve(pred[target == 1], pred[target == 0])$auc.integral)

# pred <- netAnalyze(y)

# Load used libraries
library(dplyr)
library(tidyverse)
library(RColorBrewer)
library(patchwork)
library(car)
library(ggpubr)
library(rstatix)
library(xtable)
library(comprehenr)
setwd('d:\\microbial_network\\microbial_network_explore')

# Read data
result <- read.csv("data\\temp_results\\simulation_results.csv")

result_absolute <- result %>%
  filter(abs_rel == 'Absolute') %>%
  {.}
result_relative <- result %>%
  filter(abs_rel == 'Relative') %>%
  {.}
head(result_absolute)
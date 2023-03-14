import numpy as np

from utils.compositional_lotka_volterra import CompositionalLotkaVolterra

import numpy as np
import os
from sklearn.metrics import average_precision_score
import pandas as pd
from utils.transformation import clr
from matplotlib import pyplot as plt

os.chdir('d:\\microbial_network\\microbial_network_explore')

from utils import simulation

n_vertices = 5
avg_degree = 2

time_points = 30
time_step = 0.01
downsample = 1
noise_var = 1e-3
scale_simulation = 100

network_type = 'random'
# network_type = 'small-world'

interaction_type = 'random'

adj, M = simulation.gen_graph(n_vertices, avg_degree, network_type=network_type, interaction_type=interaction_type)


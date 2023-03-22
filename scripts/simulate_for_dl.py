from utils.simulation import simulate_glv
import numpy as np
from pathlib import Path

# Definition of data: 
# abundance matrix: (n_vertices, time_points) n_vertices = 20, time_points = 50
# adj: (n_vertices, n_vertices)
# Save abundances and adj into a tuple
# Save the tuple into a list, for now 100 tuples
# Save the list into a file


path = Path('d:\\microbial_network\\microbial_network_explore\\data\\')

data = []

network_types = ['random', 'small-world', 'scale-free']

dataset_size = 100
for i in range(dataset_size):
    # with nostdout():
    z, x, y, adj, M = simulate_glv(
        num_taxa=20,
        avg_degree=np.random.randint(1, 10),
        time_points=50,
        time_step=0.1,
        downsample=1,
        noise_var=1e-3,
        network_type=np.random.choice(network_types),
        interaction_type='random',
        max_interaction_strength=1,
    )
    data.append((y, adj))

abundance, adj = zip(*data)
abundance = np.dstack(abundance)
adj = np.dstack(adj)

# Normalize abundance along the time axis
abundance = abundance / np.sum(abundance, axis=2, keepdims=True)

np.save(path / 'abundance.npy', abundance)
np.save(path / 'adj.npy', adj)


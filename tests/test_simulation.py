import numpy as np
from utils import simulation

def test_simulation():
    num_taxa = 20
    avg_degree = 5
    time_points = 1000
    downsample = 20
    z, x, y, adj, M = simulation.simulate_glv(
        num_taxa=num_taxa,
        avg_degree=avg_degree,
        time_points=time_points,
        downsample=downsample,
        seed=0,
    )
    assert z.shape == (num_taxa, time_points)
    assert x.shape == (num_taxa, time_points // downsample)
    assert y.shape == (num_taxa, time_points // downsample)
    assert adj.shape == (num_taxa, num_taxa)
    assert M.shape == (num_taxa, num_taxa)

    # Test the seed
    z2, x2, y2, adj2, M2 = simulation.simulate_glv(
        num_taxa=num_taxa,
        avg_degree=avg_degree,
        time_points=time_points,
        downsample=downsample,
        seed=0,
    )
    assert np.all(z == z2)
    assert np.all(x == x2)
    assert np.all(y == y2)
    assert np.all(adj == adj2)
    assert np.all(M == M2)

    # Test the noiseless simulation
    abundance, adj, M = simulation.simulate_noiseless_glv(
        num_taxa=num_taxa,
        avg_degree=avg_degree,
        time_points=time_points,
        downsample=downsample,
        seed=0,
    )
    assert abundance.shape == (num_taxa, time_points // downsample)
    assert adj.shape == (num_taxa, num_taxa)
    assert M.shape == (num_taxa, num_taxa)

    # Test the seed
    abundance, adj, M = simulation.simulate_noiseless_glv(
        num_taxa=num_taxa,
        avg_degree=avg_degree,
        time_points=time_points,
        downsample=downsample,
        seed=0,
    )
    assert np.all(z == z2)
    assert np.all(x == x2)
    assert np.all(y == y2)
    assert np
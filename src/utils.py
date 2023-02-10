import igraph as ig
import numpy as np

from misdeed.OmicsGenerator import OmicsGenerator

def gen_graph(n, k, network_type='random', interaction_type='random', max_interaction_strength=0.5):
    """Generate a graph and interaction matrix.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Number of edges per node.
    network_type : str, optional
        Type of graph. Choices are 'random', 'scale-free', 'small-world', and 'barabasi-albert'.
        The default is 'random'.
    interaction_type : str, optional
        Type of interaction. Choices are 'random', 'mutualism', 'competition', 'predator-prey', and 'mix'.
        The default is 'random'.
    max_interaction_strength : float, optional
        Maximum interaction strength. The default is 0.5.

    Returns
    -------
    adj : np.array
        Adjacency matrix.
    M : np.array
        Interaction matrix.

    """
    m = n * k // 2
    if network_type == 'random':
        g = ig.Graph.Erdos_Renyi(n=n, m=m)
    elif network_type == 'scale-free':
        g = ig.Graph.Static_Power_Law(n=n, m=m, exponent_out=2.2, exponent_in=-1)
    elif network_type == 'small-world':
        g = ig.Graph.Watts_Strogatz(dim=1, size=n, nei=k//2, p=0.05)
    elif network_type == 'barabasi-albert':
        g = ig.Graph.Barabasi(n=n, m=k//2)
    else:
        raise ValueError(f'Unknown network type: {network_type}')

    adj = g.get_adjacency()
    adj = np.array(adj.data)

    edge_list = g.get_edgelist()

    M = np.zeros((n, n))
    for i, j in edge_list:
        if interaction_type == 'random':
            M[i, j] = np.random.uniform(-max_interaction_strength, max_interaction_strength)
            M[j, i] = np.random.uniform(-max_interaction_strength, max_interaction_strength)
        elif interaction_type == 'mutualism':
            M[i, j] = np.random.uniform(0, max_interaction_strength)
            M[j, i] = np.random.uniform(0, max_interaction_strength)
        elif interaction_type == 'competition':
            M[i, j] = -np.random.uniform(0, max_interaction_strength)
            M[j, i] = -np.random.uniform(0, max_interaction_strength)
        elif interaction_type == 'predator-prey':
            M[i, j] = -np.random.uniform(0, max_interaction_strength)
            M[j, i] = np.random.uniform(0, max_interaction_strength)
        elif interaction_type == 'mix':
            r = np.random.uniform(0, 1)
            if r < 0.25:
                # mutualism
                M[i, j] = np.random.uniform(0, max_interaction_strength)
                M[j, i] = np.random.uniform(0, max_interaction_strength)
            elif r < 0.5:
                # competition
                M[i, j] = -np.random.uniform(0, max_interaction_strength)
                M[j, i] = -np.random.uniform(0, max_interaction_strength)
            else:
                # predator-prey
                M[i, j] = -np.random.uniform(0, max_interaction_strength)
                M[j, i] = np.random.uniform(0, max_interaction_strength)
        else:
            raise ValueError(f'Unknown interaction type: {interaction_type}')

    np.fill_diagonal(M, -1)
    return adj, M

def simulate_glv(time_points=1000, time_step=1e-2, num_taxa=100, avg_degree=10, downsample=20, noise_var=1e-2, **kwargs):
    """Simulate a GLV model.

    Parameters
    ----------
    time_points : int, optional
        Number of time points. The default is 1000.
    time_step : float, optional
        Time step. The default is 1e-2.
    num_taxa : int, optional
        Number of taxon in the network. The default is 100.
    avg_degree : int, optional
        Average degree of each taxon. The default is 10.
    downsample : int, optional
        Downsample raio. The default is 20, meaning the abundance data is record every 20-th time point.
    noise_var : float, optional
        Variance of the noise. The default is 1e-2.
    **kwargs : dict
        Keyword arguments. Passed to gen_graph().

    Returns
    -------
    z : np.array
        Absolute abundances.
    x : np.array
        Relative abundances.
    y : np.array
        Relative abundances with read noise.

    """
    adj, M = gen_graph(num_taxa, avg_degree, **kwargs)

    generator = OmicsGenerator(
        time_points=time_points,
        node_names=['mgx'],
        node_sizes=[num_taxa],
        init_full=True,
    )

    generator.add_interaction(
        name='mgx_mgx',
        outbound_node_name='mgx',
        inbound_node_name='mgx',
        matrix=M,   
    )

    z, x, y = generator.generate(
        dt=time_step,
        noise_var=noise_var,
        downsample=downsample,
    )
    z, x, y = z['mgx'], x['mgx'], y['mgx']
    return z, x, y, adj, M
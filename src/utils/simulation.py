import igraph as ig
import numpy as np
import random
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, default_converter
from misdeed.OmicsGenerator import OmicsGenerator


def gen_graph(
    n,
    k,
    network_type="random",
    interaction_type="random",
    max_interaction_strength=0.5,
    adj=None,
    M=None,
    seed=42,
):
    """Generate a graph and interaction matrix.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Average number of edges per node.
    network_type : str, optional
        Type of graph. Choices are 'random', 'scale-free', 'small-world', and 'barabasi-albert'.
        The default is 'random'.
    interaction_type : str, optional
        Type of interaction. Choices are 'random', 'mutualism', 'competition', 'predator-prey', and 'mix'.
        The default is 'random'.
    max_interaction_strength : float, optional
        Maximum interaction strength. The default is 0.5.
    adj : np.array, optional
        Adjacency matrix. If provided, the network_type argument will be ignored. The default is None.
    M : np.array, optional
        Interaction matrix. If provided, the interaction_type argument will be ignored. The default is None.
    seed : int, optional
        Random seed. The default is 42.

    Returns
    -------
    adj : np.array
        Adjacency matrix.
    M : np.array
        Interaction matrix.

    """
    random.seed(seed)
    np.random.seed(seed)

    assert n > 0 and isinstance(n, int), f"Number of nodes must be a positive integer: {n}"
    assert k > 0 and isinstance(k, int), f"Average degree for each node must be a positive integer: {k}"

    assert network_type in [
        "random",
        "scale-free",
        "small-world",
        "barabasi-albert",
    ], f"Unknown network type: {network_type}"

    assert interaction_type in [
        "random",
        "mutualism",
        "competition",
        "predator-prey",
        "mix",
    ], f"Unknown interaction type: {interaction_type}"

    if adj is None:
        m = n * k // 2  # number of edges
        if network_type == "random":
            g = ig.Graph.Erdos_Renyi(n=n, m=m)
        elif network_type == "scale-free":
            g = ig.Graph.Static_Power_Law(n=n, m=m, exponent_out=2.2, exponent_in=-1)
        elif network_type == "small-world":
            g = ig.Graph.Watts_Strogatz(dim=1, size=n, nei=k // 2, p=0.05)
        elif network_type == "barabasi-albert":
            g = ig.Graph.Barabasi(n=n, m=k // 2)
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        adj = g.get_adjacency()
        adj = np.array(adj.data)
    else:
        g = ig.Graph.Adjacency(adj.tolist())

    edge_list = g.get_edgelist()

    M = np.zeros((n, n))
    for i, j in edge_list:
        if interaction_type == "random":
            M[i, j] = np.random.uniform(-max_interaction_strength, max_interaction_strength)
            M[j, i] = np.random.uniform(-max_interaction_strength, max_interaction_strength)
        elif interaction_type == "mutualism":
            M[i, j] = np.random.uniform(0, max_interaction_strength)
            M[j, i] = np.random.uniform(0, max_interaction_strength)
        elif interaction_type == "competition":
            M[i, j] = -np.random.uniform(0, max_interaction_strength)
            M[j, i] = -np.random.uniform(0, max_interaction_strength)
        elif interaction_type == "predator-prey":
            M[i, j] = -np.random.uniform(0, max_interaction_strength)
            M[j, i] = np.random.uniform(0, max_interaction_strength)
        elif interaction_type == "mix":
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
            raise ValueError(f"Unknown interaction type: {interaction_type}")

    np.fill_diagonal(M, -1)
    return adj, M


def simulate_glv(num_taxa=20, avg_degree=10, time_points=1000, time_step=1e-2, downsample=20, noise_var=1e-2, adj=None, M=None, **kwargs):
    """Simulate a GLV model using MisDEED

    Parameters
    ----------
    num_taxa : int, optional
        Number of taxon in the network. The default is 20.
    avg_degree : int, optional
        Average degree of each taxon. The default is 10.
    time_points : int, optional
        Number of time points. The default is 1000.
    time_step : float, optional
        Time step. The default is 1e-2.
    downsample : int, optional
        Downsample raio. The default is 20, meaning the abundance data is record every 20-th time point.
    noise_var : float, optional
        Variance of the noise. The default is 1e-2.
    adj : np.array, optional
        Adjacency matrix. If provided, num_taxa, avg_degree and arguments for gen_graph() will be ignored. The default is None.
    M : np.array, optional
        Interaction strength matrix. If provided, num_taxa, avg_degree and arguments for gen_graph() will be ignored. The default is None.
    **kwargs : dict
        Keyword arguments. Passed to gen_graph().

    Returns
    -------
    z : np.array
        Absolute abundances. With shape (time_points, num_taxa).
    x : np.array
        Relative abundances. With shape (time_points//downsample, num_taxa).
    y : np.array
        Simulated read abundances with read noise. With shape (time_points//downsample, num_taxa).
    adj : np.array
        Adjacency matrix. With shape (num_taxa, num_taxa).
    M : np.array
        Interaction strength matrix. With shape (num_taxa, num_taxa).

    """

    assert num_taxa > 0 and isinstance(num_taxa, int), f"Number of taxa must be a positive integer: {num_taxa}"
    assert avg_degree > 0 and isinstance(avg_degree, int), f"Average degree must be a positive integer: {avg_degree}"
    assert time_points > 0 and isinstance(time_points, int), f"Number of time points must be a positive integer: {time_points}"
    assert time_step > 0, f"Time step must be positive: {time_step}"
    assert downsample > 0 and isinstance(downsample, int), f"Downsample ratio must be a positive integer: {downsample}"
    assert noise_var >= 0, f"Noise variance must be non-negative: {noise_var}"

    if adj is not None and M is not None:
        # Check consistency between adj and M
        assert np.array_equal(np.triu(adj, k=1), np.triu(M, k=1)!=0) and np.array_equal(np.tril(adj, k=-1), np.tril(M, k=-1)!=0), "Inconsistent adjacency matrix and interaction strength matrix"
        num_taxa = adj.shape[0]
    else:
        adj, M = gen_graph(num_taxa, avg_degree, **kwargs)

    generator = OmicsGenerator(
        time_points=time_points,
        node_names=["mgx"],
        node_sizes=[num_taxa],
        init_full=True,
    )

    generator.add_interaction(
        name="mgx_mgx",
        outbound_node_name="mgx",
        inbound_node_name="mgx",
        matrix=M,
    )

    z, x, y = generator.generate(
        dt=time_step,
        noise_var=noise_var,
        # downsample=downsample, down sample is deprecated
    )
    z, x, y = z["mgx"], x["mgx"], y["mgx"]
    # z, x, y = z.T, x.T, y.T
    z = z[::downsample]
    x = x[::downsample]
    y = y[::downsample]
    return z, x, y, adj, M


def simulate_noiseless_glv(num_taxa=20, avg_degree=10, time_points=1000, downsample=20, adj=None, M=None, **kwargs):
    """Simulate a noiseless GLV model with the R package seqtime

    Parameters
    ----------
    num_taxa : int, optional
        Number of taxon in the network. The default is 20.
    avg_degree : int, optional
        Average degree of each taxon. The default is 10.
    time_points : int, optional
        Number of time points. The default is 1000.
    downsample : int, optional
        Downsample raio. The default is 20, meaning the abundance data is record every 20-th time point.
    adj : np.array, optional
        Adjacency matrix. If provided, num_taxa, avg_degree and arguments for gen_graph() will be ignored. The default is None.
    M : np.array, optional
        Interaction strength matrix. If provided, num_taxa, avg_degree and arguments for gen_graph() will be ignored. The default is None.
    **kwargs : dict
        Keyword arguments. Passed to gen_graph().

    Returns
    -------
    abundance : np.array
        Absolute abundances. With shape (time_points//downsample, num_taxa).
    adj : np.array
        Adjacency matrix. With shape (num_taxa, num_taxa).
    M : np.array
        Interaction strength matrix. With shape (num_taxa, num_taxa).

    """

    assert num_taxa > 0 and isinstance(num_taxa, int), f"Number of taxa must be a positive integer: {num_taxa}"
    assert avg_degree > 0 and isinstance(avg_degree, int), f"Average degree must be a positive integer: {avg_degree}"
    assert time_points > 0 and isinstance(time_points, int), f"Number of time points must be a positive integer: {time_points}"
    assert downsample > 0 and isinstance(downsample, int), f"Downsample ratio must be a positive integer: {downsample}"

    if adj is not None and M is not None:
        # Check consistency between adj and M
        assert np.array_equal(np.triu(adj, k=1), np.triu(M, k=1)!=0) and np.array_equal(np.tril(adj, k=-1), np.tril(M, k=-1)!=0), "Inconsistent adjacency matrix and interaction strength matrix"
        num_taxa = adj.shape[0]
    else:
        adj, M = gen_graph(num_taxa, avg_degree, **kwargs)

    # Load seqtime package in R if not loaded
    try:
        seqtime
    except NameError:
        seqtime = importr("seqtime")

    nv_cv_rules = default_converter+ numpy2ri.converter
    with nv_cv_rules.context():
        abundance = seqtime.generateDataSet(time_points, M, count=num_taxa*10, mode=4)

    abundance = abundance[:, ::downsample]
    abundance = abundance.T

    return abundance, adj, M


def test_seed():
    np.random.seed(42)
    print(np.random.rand(10))

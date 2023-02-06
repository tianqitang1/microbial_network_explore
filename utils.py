import igraph as ig

def gen_graph(n, k, network_type='random', interaction_type='random', max_interaction_strength=0.5, max_competition_strength=0.5):
    m = n * k // 2
    if network_type == 'random':
        g = ig.Graph.Erdos_Renyi(n=n, m=m)
    elif network_type == 'scale-free':
        g = ig.Graph.Static_Power_Law(n=n, m=m, exponent_out=2.2, exponent_in=-1)
    elif network_type == 'small-world':
        g = ig.Graph.Watts_Strogatz(dim=1, size=n, nei=k//2, p=0.05)
    elif network_type == 'barabasi-albert':
        g = ig.Graph.Barabasi(n=n, m=k//2)
    return g
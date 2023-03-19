import numpy as np
import os
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd
from utils.transformation import clr_transform, alr_transform
from matplotlib import pyplot as plt
from itertools import product
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from utils import simulation
import rpy2.robjects as robjects
import seaborn as sns
from utils.generalized_lotka_volterra import GeneralizedLotkaVolterra
from utils.compositional_lotka_volterra import CompositionalLotkaVolterra
from scipy.stats import ttest_rel
from typing import List
from utils.evaluations import correlation_score, precision_matrix_score, clv_score, glv_score, pcor_score, sparcc_score, speic_score, calc_nondiag_score
import seaborn as sns
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests


os.chdir('d:\\microbial_network\\microbial_network_explore')


def evaluation(adj, abundance, evaluation_func, metrics=average_precision_score, verbose=False):
    scores = []
    for func in evaluation_func:
        scores.append([func._method, *func(abundance, adj, metrics=metrics, verbose=verbose)])
    columns = ['Method']
    columns.extend([metric.__name__ for metric in metrics] if isinstance(metrics, List) else [metrics.__name__])
    scores_df = pd.DataFrame(scores, columns=columns)
    return scores_df


def main():
    n_vertices = 5
    avg_degree = 2

    time_points = 30
    time_step = 0.01
    downsample = 1
    noise_var = 1e-3
    scale_simulation = 100

    # n_vertices = 50
    # avg_degree = 5

    # time_points = 100
    # time_step = 0.01
    # downsample = 1
    # noise_var = 1e-3
    # scale_simulation = 100

    network_type = 'random'
    # network_type = 'small-world'

    interaction_type = 'random'
    
    normal_noise_df = pd.DataFrame(columns=['Method', 'average_precision_score', 'roc_auc_score', 'n_vertices', 'avg_degree', 'noise_var'])

    n_vertices = [20, 50, 100]
    avg_degree = [2, 5, 10]
    noise_var = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1,]

    evaluation_func = [correlation_score, precision_matrix_score, clv_score, glv_score, pcor_score, sparcc_score, speic_score]
    metrics = [average_precision_score, roc_auc_score]

    # nonoise_df = pd.DataFrame(columns=['Method', 'average_precision_score', 'roc_auc_score', 'n_vertices', 'avg_degree', 'noise_var'])
    # for n, k in product(n_vertices, avg_degree):
    #     adj, M = simulation.gen_graph(n, k, network_type=network_type, interaction_type=interaction_type)
    #     abd, _, _ = simulation.simulate_noiseless_glv(time_points=time_points, downsample=downsample, adj=adj, M=M)
    #     scores_df = evaluation(adj, abd, evaluation_func, metrics=metrics, verbose=False)
    #     scores_df['n_vertices'] = n
    #     scores_df['avg_degree'] = k
    #     scores_df['noise_var'] = 0
    #     nonoise_df = pd.concat([nonoise_df, scores_df], axis=0)
        
    #     for s in noise_var:
    #         abd = abd * (1 + np.random.normal(0, s, abd.shape))

    #         scores_df = evaluation(adj, abd, evaluation_func, metrics=metrics, verbose=False)
    #         scores_df['n_vertices'] = n
    #         scores_df['avg_degree'] = k
    #         scores_df['noise_var'] = s
    #         normal_noise_df = pd.concat([normal_noise_df, scores_df], axis=0)
    # nonoise_df.to_csv('nonoise.csv', index=False)
    # normal_noise_df.to_csv('normal_noise.csv', index=False)

    misdeed_df = pd.DataFrame(columns=['Method', 'average_precision_score', 'roc_auc_score', 'n_vertices', 'avg_degree', 'noise_var'])
    for n, k, s in product(n_vertices, avg_degree, noise_var):
        adj, M = simulation.gen_graph(n, k, network_type=network_type, interaction_type=interaction_type)

        z, x, abundance, _, _ = simulation.simulate_glv(time_points=time_points, downsample=downsample, adj=adj, M=M, noise=noise_var)

        scores_df = evaluation(adj, abundance, evaluation_func, metrics=metrics, verbose=False)
        scores_df['n_vertices'] = n
        scores_df['avg_degree'] = k
        scores_df['noise_var'] = s
        misdeed_df = pd.concat([normal_noise_df, scores_df], axis=0)
    
    misdeed_df.to_csv('misdeed.csv', index=False)
    # sns.barplot(x='n_vertices', y='roc_auc_score', hue='method', data=scores_df_all)
    # plt.show()


if __name__ == '__main__':
    main()
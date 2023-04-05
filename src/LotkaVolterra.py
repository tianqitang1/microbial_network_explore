import numpy as np
import sys
from scipy.special import logsumexp
from scipy.stats import linregress
from scipy.integrate import RK45, solve_ivp
from utils.transformation import clr_transform, alr_transform


class CompositionalLotkaVolterra:
    def __init__(self) -> None:
        pass

    @staticmethod
    def solve_clv(X, )
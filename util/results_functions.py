import jax.numpy as np
import matplotlib.pyplot as plt
from IPython import display
from toolz.dicttoolz import valmap, itemmap
from itertools import chain

def report_ave_costs(costss, Tc=None):
    T = len(costss)
    def ave_cost(x):
        return np.sum(np.array(x)) / len(x)

    ave_sum = lambda x: np.sum(np.array(x)) / len(x)

    if Tc is None:
        ave_sums = valmap(ave_sum, costss)
        for Cstr, costs in ave_sums.items():
            print("The averege cost by", Cstr, ave_sums[Cstr])

    else:
        for Cstr, costs in costss.items():
            print("The averege cost for k <", Tc, "by", Cstr, ave_cost(costs[:Tc]))
            print("The averege cost for k >", Tc, "by", Cstr, ave_cost(costs[Tc:]))

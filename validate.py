import arviz as az

from .definitions import *
from .modeller import Modeller
from .sales_model import SalesModel 

import matplotlib.pyplot as plt


def CompareModels(*args):
    models_dict = {m.Name(): m.ToArviZ() for m in args}
    compare_results = az.compare(models_dict, ic='loo')
    print(compare_results)
    az.plot_compare(compare_results)
    #az.plot_forest(list(models_dict.values()), var_names=['beta'])
    plt.show()
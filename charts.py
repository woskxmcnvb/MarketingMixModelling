from numpyro.diagnostics import hpdi

import matplotlib.pyplot as plt
import seaborn as sns

from .definitions import *


def PlotPrediction(y_prediction: jnp.array, y_observed=None, x_axis=None, conf_band=True, conf_level=0.9):
    mid = y_prediction.mean(axis=0)
    if x_axis is None: 
        x_axis = range(len(mid))
    
    if conf_band:
        lb, ub = hpdi(y_prediction, conf_level)
        plt.fill_between(x=x_axis, y1=lb.ravel(), y2=ub.ravel(), color="green", alpha=0.2)

    plt.plot(x_axis, mid, color='red')
    if y_observed is not None:
        plt.plot(x_axis, y_observed, color='grey', linestyle='', marker='o', markersize=5, alpha=0.4)


"""def PlotX(X: ModelCovs): 
        tiles_num = len(X.media_vars) + len(X.non_media_vars)
        _, axs = plt.subplots(tiles_num, 1, figsize=(16, tiles_num * 2))

        next_tile = 0
        for var in X.media_vars:
            pd.DataFrame(X.media_data[:, var.data_index], 
                         columns=var.VarColumns()).plot.area(ax=axs[next_tile], linewidth=0, stacked=False)
            next_tile = next_tile + 1
        for var in X.non_media_vars:
            pd.DataFrame(X.non_media_data[:, var.data_index], 
                         columns=var.VarColumns()).plot.line(ax=axs[next_tile])
            next_tile = next_tile + 1
        plt.show()"""
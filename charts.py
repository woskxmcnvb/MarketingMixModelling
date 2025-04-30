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

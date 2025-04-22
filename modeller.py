import io

from copy import deepcopy

import pandas as pd
import numpy as np 
import jax.numpy as jnp
import numpyro

from .smoother import Smoother
from .sales_model import SalesModel

import matplotlib.pyplot as plt
import seaborn as sns

from .definitions import *

def AreaChartWithNegative(df, ax, ylim=None):
    area_positive = df.where(df > 0, 0)
    area_negative = df.where(df < 0, 0)
    
    if ylim is None:
        ylim = (
            1.10 * area_negative.sum(axis=1).min(axis=None) if area_negative is not None else 0,
            1.05 * area_positive.sum(axis=1).max(axis=None) if area_positive is not None else 0
        )
    if area_positive is not None:
        area_positive.plot.area(ax=ax, linewidth=0, ylim=ylim, color = sns.color_palette("muted"))##, len(area_negative.columns)))
    h, _ = ax.get_legend_handles_labels()
    if area_negative is not None:
        area_negative.plot.area(ax=ax, linewidth=0, ylim=ylim, color = sns.color_palette("muted"))

    ax.legend(handles=h)

def PlotX(X: ModelCovs): 
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
        plt.show()
            


class Modeller:
    # data
    spec: dict
    input_df: pd.DataFrame
    X: ModelCovs = None
    y: ModelTarget = None

    # result
    model = None
    decomposition: pd.DataFrame = None

    def __init__(self):
        pass

    def PrepNoFit(self, spec: dict, data: pd.DataFrame):
        self.input_df = data
        self.spec = spec.copy()
        self.X = ModelCovs(spec=self.spec).FitToData(data)
        self.y = ModelTarget().Fit(self.spec, data)
        return self
    
    def PrepareNewCovs(self, df: pd.DataFrame) -> ModelCovs:
        return self.X.Copy().TransformData(df)
    
    def Fit(self, spec: dict, data: pd.DataFrame, show_progress=True, num_samples=1000):
        self.PrepNoFit(spec, data)
        self.model = SalesModel(
            seasonality_spec=self.X.seasonality, 
            fixed_base=self.X.fixed_base, 
            long_term_retention=self.X.long_term_retention
        ).Fit(self.X, self.y, show_progress=show_progress, num_samples=num_samples)
        return self
    
    def Predict(self, data: pd.DataFrame) -> dict:
        return self.model.Predict(self.PrepareNewCovs(data))

    def GetDecomposition(self):
        if self.decomposition is not None: 
            return self.decomposition
        
        sample = self.model.Predict(self.X)

        col_names = {
            "y": [("y", "y")],
            "base": [("base", "base")],
        }
        if self.X.HasMedia():
            col_names["media short"] = [("Media short", v) for _, v in self.X.AllMediaVarnames()]
            col_names["media long"] = [("Media long", v) for _, v in self.X.AllMediaVarnames()]
        if self.X.HasNonMedia():
            col_names["non-media"] = [("Non-media", v) for _, v in self.X.AllNonMediaVarnames()]

        col_names = {k: pd.MultiIndex.from_tuples(v) for k, v in col_names.items()}

        decomposition = pd.concat(
            [pd.DataFrame(v.mean(axis=0), columns=col_names[k]) for k, v in sample.items() if k in col_names], 
            axis=1
        ).set_axis(self.input_df.index, axis=0)
        self.decomposition = self.y.scaler.InverseTransform(decomposition)
        return self.decomposition
    
    def GetSamples(self):
        return self.model.mcmc.get_samples()
    
    def SitesNames(self):
        return self.model.mcmc.get_samples().keys()
    
    










    ########################## *********CHARTS *****************

    def PlotInputs(self): 
        tiles_num = len(self.X.media_vars) + len(self.X.non_media_vars) + 1
        _, axs = plt.subplots(tiles_num, 1, figsize=(16, tiles_num * 2))
        if tiles_num == 1:
            axs.plot(self.y.y, label='Dependent')
            axs.legend()
        else:
            axs[0].plot(self.y.y, label='Dependent')
            axs[0].legend()

        next_tile = 1
        for var in self.X.media_vars:
            pd.DataFrame(self.X.media_data[:, var.data_index], 
                         columns=var.VarColumns()).plot.area(ax=axs[next_tile], linewidth=0, stacked=False)
            next_tile = next_tile + 1
        for var in self.X.non_media_vars:
            pd.DataFrame(self.X.non_media_data[:, var.data_index], 
                         columns=var.VarColumns()).plot.line(ax=axs[next_tile])
            next_tile = next_tile + 1
        plt.show()




    def PlotFit(self):
        decomposition = self.GetDecomposition()
        
        fig, axs = plt.subplots(1, 1, figsize=(16, 6))
        decomposition["y"].set_axis(['Modelled metric'], axis=1).plot.line(color='red', ax=axs)
        self.input_df[self.spec["y"]].plot.line(ax=axs, label="Observed metric", color='grey', alpha=0.5)
        axs.legend()


    def PlotDecomposition(self, decomp_spec: dict, media_spec: dict=None, ylim: tuple=None):
        decomposition = self.GetDecomposition()
        chart_data = pd.concat({k: decomposition[v].sum(1) for k, v in decomp_spec.items()}, axis=1)

        if (media_spec is None) or (len(media_spec)==0):
            _, axs = plt.subplots(1, 1, figsize=(16, 6))
            AreaChartWithNegative(chart_data, axs, ylim=ylim)

        else:
            _, axs = plt.subplots(len(media_spec)+1, 1, figsize=(16, 6+len(media_spec)), height_ratios=[7]+[1]*len(media_spec))
            AreaChartWithNegative(chart_data, axs[0], ylim=ylim)

            for tile, (_, columns) in enumerate(media_spec.items()):
                pd.concat(
                    {name: self.input_df[cols].sum(1) for name, cols in columns.items()}, axis=1
                ).plot.area(ax=axs[tile+1], linewidth=0, stacked=False)

    ########################## *********CHARTS END*****************

































    def PlotSiteIntervals(self, site: str, interval: float=0.5, center: str='median', names: list=None, title: str=None, ax=None):
        """
        center in ['mean', 'median', 'mid_hpdi']
        """
        sample = self.GetSamples()[site]
        hpdi_ = numpyro.diagnostics.hpdi(sample, interval)
        if center == 'mean':
            center_ = sample.mean(0)
        elif center == 'median':
            center_ = np.median(sample, axis=0)
        elif center == 'mid_hpdi':
            center_ = hpdi_.mean(0)
        else:
            raise ValueError("PlotSiteIntervals: check center spec")
        hpdi_ = numpyro.diagnostics.hpdi(sample, interval)
        data_for_plot = pd.DataFrame(np.column_stack([hpdi_[0], center_, hpdi_[1]]))
        if names: 
            data_for_plot.index = names
        IntervalPlot(data_for_plot, ax=ax, title=title)

    def PlotDiminishingReturnCurves(self, hpdi=0.5): 
        if not self.diminishing_return: 
            print("Not a diminishing return model")
            return
        if MEDIA_OWN not in self.spec['X']: 
            print("No own media in the model") 
        sample = self.GetSamples()
        #alpha = sample['alpha'].mean(0)
        #gamma = sample['gamma'].mean(0)
        alpha = np.median(sample['alpha'], axis=0)
        gamma = np.median(sample['gamma'], axis=0)
        x = np.linspace(0, 1, 20)
        x_pow_a = np.power(np.repeat(x[:, np.newaxis], len(alpha), axis=1), alpha)
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        pd.DataFrame(x_pow_a / (x_pow_a + np.power(gamma, alpha)), index=x, columns=self.spec['X'][MEDIA_OWN]).plot.line(ax=axs[0])
        self.PlotSiteIntervals('alpha', center='median', ax=axs[1], title='Alpha')
        self.PlotSiteIntervals('gamma', center='median', ax=axs[2], title='Gamma')
    

        

    def PlotNonMediaDecomposition(self, ylim: tuple=None):
        if self.decomposition is None:
            self.GetDecomposition()
        
        _, ax = plt.subplots(1, 1, figsize=(16, 6))
        AreaChartWithNegative(
            self.decomposition[self.__GetNonMediaSites(include_seasonality=False)],  #.T.groupby(level=0).sum().T, 
            ax, ylim=ylim
        )
        
        
    def PlotMediaDecomposition(self):
        if not MEDIA_OWN in self.X:
            return 
        fig, axs = plt.subplots(3, 1, figsize=(16, 6), height_ratios=[7, 1, 1])
        
        self.decomposition[MEDIA_OWN].plot.area(ax=axs[0], linewidth=0, stacked=False)
        self.input_df[self.spec['X'][MEDIA_OWN]].plot.area(ax=axs[1], linewidth=0, stacked=False)
        if MEDIA_COMP in self.X:
            self.input_df[self.spec['X'][MEDIA_COMP]].plot.area(ax=axs[2], linewidth=0, stacked=False)

        









def IntervalPlot(data: pd.DataFrame, ax=None, title=None):
    """
    принимает 3 столбца: первый, последний - края, средний - точка
    строки = категории
    """
    if ax is None: 
        ax = plt
        if title:
            plt.title(title)
    else: 
        if title:
            ax.set_title(title)
    for _, row in data.iterrows():
        ax.plot((row.iloc[0], row.iloc[-1]), (row.name, row.name))
        ax.scatter(row.iloc[1], row.name)
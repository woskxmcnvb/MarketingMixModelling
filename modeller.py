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

#from utils import FixDirPath

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
            

class ModelInputs:
    media_vars: list[VariableGroup]
    media_data: jnp.array 
    
    non_media_vars: list[VariableGroup]
    non_media_data: jnp.array

    seasonality: dict = None
    fixed_base: bool = False
    long_term_retention: int | tuple = 1

    y: jnp.array
    scalers: dict[Scaler]

    def __init__(self, spec: dict, df: pd.DataFrame):
        self.media_vars = []
        self.non_media_vars = []
        self.scalers = dict()
        
         # check must-keys
        missing_keys = []
        for mk in ["name", "fixed base", "long-term retention", "y", "X"]:
            if mk not in spec:
                missing_keys.append(mk)
        assert len(missing_keys) == 0, "ERRROR! Check missing keys in spec {}.".format(missing_keys)

        # check fixed base option
        assert isinstance(spec["fixed base"], bool), "Wrong 'fixed base' value, bool is expected"
        self.fixed_base = spec["fixed base"]

        if isinstance(spec["long-term retention"], tuple) or isinstance(spec["long-term retention"], list):
            self.long_term_retention = list(spec["long-term retention"])
        elif spec["long-term retention"] == 1:
            self.long_term_retention = 1 
        else:
            raise ValueError("Wrong 'long-term retention' value, 1 or list or tuple[int, int] is expected")
        
        # check fix seasonality 
        if "seasonality" in spec:
            assert "cycle" in spec["seasonality"], "Seasonality cycle must be specified"
            period = spec["seasonality"]["cycle"]
            assert period is not None, "Seasonality period must be specified"
            assert 1 < period and period <= 52, "Wrong seasonality cycle: {}".format(period)

            assert "model" in spec["seasonality"], "Seasonality model must be specified"
            model = spec["seasonality"]["model"]
            assert model is not None, "Seasonality model must be specified"
            assert model in ['fourier', 'discrete'], "Wrong seasonality model: {}".format(model)

            self.seasonality = spec["seasonality"].copy()
            
            if model == 'fourier':
                if ("num_fouries_terms" not in self.seasonality) or (self.seasonality["num_fouries_terms"] is None): 
                    self.seasonality["num_fouries_terms"] = period // 4

        # prepare y
        assert isinstance(spec["y"], str), "Wrong 'y' format in spec"
        self.scalers['y'] = Scaler(scaling='max_only', scaler_from='column').Fit(df[spec["y"]])
        self.y = jnp.array(Smoother().Impute(self.scalers['y'].Transform(df[spec["y"]])))

        # prepare X
        _media_data = []
        _media_data_index = 0
        _non_media_data = []
        _non_media_data_index = 0
        for var_group in spec["X"]:
            if var_group["type"] == 'media':
                self.media_vars.append(VariableGroup(self).FromDict(var_group))
                dims = self.media_vars[-1].Dims()
                self.media_vars[-1].SetDataIndex(_media_data_index, _media_data_index + dims)
                _media_data_index += dims
                _media_data.append(self.media_vars[-1].PrepareMediaData(df))
            if var_group["type"] == 'non-media':
                self.non_media_vars.append(VariableGroup(self).FromDict(var_group))
                dims = self.non_media_vars[-1].Dims()
                self.non_media_vars[-1].SetDataIndex(_non_media_data_index, _non_media_data_index + dims)
                _non_media_data_index += dims
                _non_media_data.append(self.non_media_vars[-1].PrepareNonMediaData(df))
            else: 
                Warning("Wrong X variable type: {}. Skipped".format(var_group["type"]))
        
        self.media_data = jnp.column_stack(_media_data)
        self.non_media_data = jnp.column_stack(_non_media_data)
    
    def Copy(self):
        return deepcopy(self)



class Modeller:
    # data
    spec: dict
    input_df: pd.DataFrame
    model_inputs: ModelInputs = None

    # result
    model = None
    decomposition: pd.DataFrame = None

    def __init__(self):
        pass

    def PrepNoFit(self, spec: dict, data: pd.DataFrame):
        self.input_df = data
        self.spec = spec
        self.model_inputs = ModelInputs(spec=spec, df=data)
        return self
        
    
    def Fit(self, spec: dict, data: pd.DataFrame, show_progress=True, num_samples=1000):
        self.PrepNoFit(spec, data)
        self.model = SalesModel(
            seasonality_spec=self.seasonality, 
            fixed_base=self.spec["fixed base"], 
            long_term_retention=self.spec["long-term retention"]
        ).Fit(X=self.X, y=self.y, show_progress=show_progress, num_samples=num_samples)
        return self

    def GetDecomposition(self):
        if self.decomposition is not None: 
            return self.decomposition
        
        sample = self.model.SampleModel(self.X)

        col_names = {
            "y": [("y", "y")],
            "base": [("base", "base")],
        }
        if self.X["media"]:
            col_names["media"] = sum([g.VarNamesAsTuples() for g in self.X["media"]], [])
            col_names["media long"] = sum([g.VarNamesAsTuples(suffix = " long") for g in self.X["media"]], [])
        if self.X["non-media"]:
            col_names["non-media"] = sum([g.VarNamesAsTuples() for g in self.X["non-media"]], [])

        col_names = {k: pd.MultiIndex.from_tuples(v) for k, v in col_names.items()}

        decomposition = pd.concat(
            [pd.DataFrame(v.mean(axis=0), columns=col_names[k]) for k, v in sample.items() if k in col_names], 
            axis=1
        ).set_axis(self.input_df.index, axis=0)
        self.decomposition = self.scalers['y'].InverseTransform(decomposition)
        return self.decomposition
    
    def GetSamples(self):
        return self.model.mcmc.get_samples()
    
    def SitesNames(self):
        return self.model.mcmc.get_samples().keys()
    
    


























    ########################## *********CHARTS *****************

    def PlotInputs(self): 
        tiles_num = len(self.model_inputs.media_vars) + len(self.model_inputs.non_media_vars) + 1
        _, axs = plt.subplots(tiles_num, 1, figsize=(16, tiles_num * 2))
        if tiles_num == 1:
            axs.plot(self.model_inputs.y, label='Dependent')
            axs.legend()
        else:
            axs[0].plot(self.model_inputs.y, label='Dependent')
            axs[0].legend()

        next_tile = 1
        for var in self.model_inputs.media_vars:
            pd.DataFrame(var.GetData(), 
            #pd.DataFrame(self.model_inputs.media_data[:, var.data_index], 
                         columns=var.VarColumns()).plot.area(ax=axs[next_tile], linewidth=0, stacked=False)
            next_tile = next_tile + 1
        for var in self.model_inputs.non_media_vars:
            pd.DataFrame(var.GetData(), 
            #pd.DataFrame(self.model_inputs.non_media_data[:, var.data_index], 
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
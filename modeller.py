import io

import pandas as pd
import numpy as np 
import numpyro

from smoother import Smoother
from sales_model import SalesModel

import matplotlib.pyplot as plt
import seaborn as sns

#from utils import FixDirPath

from definitions import *

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


class Scaler:
    """
    scaling in ['min_max', 'max_only']
    mode in ['total', 'column']
    centering in [None, 'mean', 'first']
    """
    min_ = None
    max_ = None
    centering_shift = 0
    fit_shape_ = None
    scaling: str
    centering: str = None
    scaler_from: str

    def __init__(self, scaling='min_max', scaler_from='column', centering=None):
        # scaling in 'min_max' / 'max_only'
        assert scaler_from in ['total', 'column']
        assert scaling in ['min_max', 'max_only']
        assert centering in [None, 'mean', 'first']
        self.scaling = scaling
        self.centering = centering
        self.scaler_from = scaler_from

    def Inspect(self):
        print("min: {}, max: {}, scaling: {}, centeing: {}".format(self.min_, self.max_, self.scaling, self.centering))

    def Fit(self, data: pd.DataFrame):
        self.fit_shape_ = data.shape
        axis_ = 0 if self.scaler_from == 'column' else None
        self.min_ = data.min(axis=axis_) if self.scaling == 'min_max' else 0
        self.max_ = data.max(axis=axis_)
        if self.centering:
            transformed = self.Transform(data)
            if self.centering == 'first':
                self.centering_shift = transformed.iloc[0]
            elif self.centering == 'mean':
                self.centering_shift = transformed.mean(axis=0)
        return self 

    def Transform(self, data: pd.DataFrame):
        #assert data.shape == self.fit_shape_
        return data.sub(self.min_).div(self.max_ - self.min_).sub(self.centering_shift)
    
    def FitTransform(self, data: pd.DataFrame):
        return self.Fit(data).Transform(data)
    
    def InverseTransform(self, data):
        #assert data.shape == self.fit_shape_
        return data.add(self.centering_shift).mul(self.max_ - self.min_).add(self.min_)

spec_format = [
    {
        "name": "Named variable group",  # обязательное поле, str
        "type": "media",                 # обязательное поле варианты ["media", "media"]
        "scaling": "total",              # обязательно для "type"=="non-media", варианты ["total", "column"]
        "saturation": True,              # обязательно для "type"=="media", для "type"=="non-media" не используется, bool 
        "variables": [
            {
                "name": "TV",            # обязательное поле, str
                "column": "CH TV",       # обязательное поле, str
                "rolling": 3,            # обязательное поле для "type"=="media", int
                "retention": (3, 1),     # обязательное поле для "type"=="media", tuple (3, 1)
                "beta": 1,               # ---
                "force_positive": True   # --- 
            },
        ]
    }, 
]

class VariableGroup:
    spec: dict = None
    name: str = None
    data: np.array = None
    
    def __init__(self):
        pass

    def CheckSpec(self, spec): 
        for field in ["name", "type", "variables"]:
            assert field in spec, "Missing key '{}' in {}".format(field, spec["name"])
        
        assert isinstance(spec["name"], str), "Wrong 'name' format in {}".format(spec["name"])
        
        assert spec["type"] in ["media", "non-media"], "Wrong 'type' {} in {}".format(spec["type"], spec["name"])
        
        if spec["type"] == "non-media":
            assert "scaling" in spec, "Missing 'scaling' key in {}".format(spec["name"])
        if "scaling" in spec:
            assert spec["scaling"] in ["total", "column"], "Wrong 'scaling' {} in {}".format(spec["scaling"], spec["name"])

        if spec["type"] == "media":
            assert "saturation" in spec, "Missing 'saturation' key in {}".format(spec["name"])
            assert isinstance(spec["saturation"], bool), "Wrong 'saturation' format in {}".format(spec["name"])

        assert isinstance(spec["variables"], list), "Wrong 'variables' format in {}".format(spec["name"])
        for var in spec["variables"]:
            self.CheckVarSpec(var, spec)
    
    def CheckVarSpec(self, var, spec):
        assert isinstance(var, dict), "Variable format is not dict in {}".format(spec["name"])
        for field in ["name", "column"]:
            assert field in var, "Missing key '{}' in {}".format(field, spec["name"])
        
        assert isinstance(var["name"], str), "Wrong 'name' format in {}".format(spec["name"])
        assert isinstance(var["column"], str), "Wrong 'column' format in {}".format(spec["name"])

        if spec["type"] == "media":
            assert "rolling" in var, "Missing 'rolling' key in {}".format(spec["name"])
            assert isinstance(var["rolling"], int) and var["rolling"] > 0, "Wrong 'rolling' value in {}".format(spec["name"])
        
        if spec["type"] == "media":
            assert "retention" in var, "Missing 'retention' key in {}".format(spec["name"])
            assert isinstance(var["retention"], tuple) and len(var["retention"]) == 2, "Wrong 'retention' value in {}".format(spec["name"])

    def FromDict(self, spec: dict):
        self.CheckSpec(spec)
        self.spec = spec
        return self 
    
    def Name(self):
        return self.name

    def Type(self):
        return self.spec["type"]
    
    def VarNames(self): 
        assert self.spec is not None, "Initiate first"
        return [v["name"] for v in self.spec["variables"]]
    
    def VarColumns(self):
        assert self.spec is not None, "Initiate first"
        return [v["column"] for v in self.spec["variables"]]
    
    def ForceVector(self):
        assert self.spec is not None, "Initiate first"
        return [int(v["force_positive"]) for v in self.spec["variables"]]
    
    def BetaVector(self):
        assert self.spec is not None, "Initiate first"
        return [v["beta"] for v in self.spec["variables"]]
    
    def RetentionVector(self):
        assert self.spec is not None, "Initiate first"
        return [v["retention"][0] for v in self.spec["variables"]], [v["retention"][1] for v in self.spec["variables"]]
    
    def RollingDict(self): 
        # если четное делает +1 
        assert self.spec is not None, "Initiate first"
        def _to_odd(x):
            return x + 1 if x % 2 == 0 else x
        return {v["column"]: _to_odd(v["rolling"]) for v in self.spec["variables"]}
    
    def PrepareData(self, df: pd.DataFrame):
        assert self.spec is not None, "Initiate first"
        if self.spec["type"] == "media":
            rolling_dict = self.RollingDict()
            self.data = df[self.VarColumns()]\
                    .where(df[self.VarColumns()] > 0)\
                    .apply(lambda x: x.rolling(window=rolling_dict[x.name], min_periods=1, center=True).mean())\
                    .where(df[self.VarColumns()] > 0)\
                    .fillna(0)\
                    .pipe(lambda x: x.div(x.max(axis=None)))\
                    .values
        elif self.spec["type"] == "non-media":
            self.data = Smoother().Impute(
                Scaler(
                    scaling='min_max', 
                    scaler_from=self.spec["scaling"], 
                    centering='first').FitTransform(df[self.VarColumns()])
            )
        return self
    
    def GetData(self):
        return self.data
            



class Modeller:
    """
    seasonality_period: int=1 //1 mean no seasonality used
    seasonality_model in ['discrete', 'fourier']
    seasonality_num_fouries_terms: int=None   // если не специфицировано будет seasonality_period / 4
    """

    # data
    input_df: pd.DataFrame
    X: dict = None
    y: np.array
    scalers: dict[Scaler]

    # result
    model = None
    decomposition: pd.DataFrame = None
    base_sites: list
    struct_sites: list

    def __init__(self):
        self.X = {
            "media": [],
            "non-media": []
        }
        self.scalers = dict()

    def PrepNoFit(self, spec: dict, data: pd.DataFrame, show_progress=True):
        # to check inputs
        self.input_df = data
        self.spec = spec
        self.__PrepareInputs()
        return self 

    def __PrepareInputs(self):
        # prepare y
        assert isinstance(self.spec["y"], str), "Wrong 'y' format in spec"
        df = self.input_df[self.spec["y"]]
        self.scalers['y'] = Scaler(scaling='max_only').Fit(df)
        self.y = Smoother().Impute(self.scalers['y'].Transform(df))

        # prepare X
        for var_group in self.spec["X"]:
            self.X[var_group["type"]].append(
                VariableGroup().FromDict(var_group).PrepareData(self.input_df)
            )
        
    
    """def __init__(self, 
                 model_name,
                 #diminishing_return: bool=True,
                 #media_retention_factor: int=3, 
                 seasonality_period: int=1, 
                 seasonality_model='discrete', 
                 seasonality_num_fouries_terms: int=None) -> None:
        assert seasonality_model in ['discrete', 'fourier']

        self.diminishing_return = diminishing_return
        self.media_retention_factor = media_retention_factor
        self.seasonality_period = seasonality_period
        self.seasonality_model = seasonality_model
        if seasonality_num_fouries_terms is not None:
            self.seasonality_num_fouries_terms = seasonality_num_fouries_terms
        else:
            self.seasonality_num_fouries_terms = self.seasonality_period / 4
        self.model_name = model_name"""
    
    
    
    def Fit(self, data: pd.DataFrame, spec: dict, show_progress=True, num_samples=1000):
        self.input_df = data
        self.spec = self.CheckFixSpec(spec)
        self.PrepareInputs()
        
        self.model = SalesModel(
            diminishing_return=self.diminishing_return,
            seasonality_period=self.seasonality_period, 
            seasonality_model=self.seasonality_model,
            seasonality_num_fouries_terms=self.seasonality_num_fouries_terms,
            retention_factor=self.media_retention_factor
        ).Fit(X=self.X, y=self.y, show_progress=show_progress, num_samples=num_samples)

        return self

    def GetDecomposition(self):
        sample = self.model.SampleModel(self.X)
        sample_len = sample['base'].shape[-1]
        
        decomposition = {}
        for k, v in sample.items():
            if k == MODEL_MEDIA:
                all_media = pd.DataFrame(v.mean(axis=0), 
                    columns=sum([self.spec['X'][m] for m in [MEDIA_OWN, MEDIA_OWN_LOW_RET, MEDIA_COMP] if m in self.spec['X']], []))
                for m in [MEDIA_OWN, MEDIA_OWN_LOW_RET, MEDIA_COMP]: 
                    if m in self.spec['X']:
                        decomposition[m] = all_media.loc[:, self.spec['X'][m]]
            elif k == MODEL_SEASONAL:
                decomposition[k] = pd.DataFrame(
                    np.tile(v.squeeze(-1).mean(axis=0), int(np.ceil(sample_len / self.seasonality_period)))[:sample_len, np.newaxis], 
                    columns=[MODEL_SEASONAL])
            else:
                decomposition[k] = pd.DataFrame(
                    v.mean(axis=0), 
                    columns=(self.spec['X'][k] if k in self.spec['X'] else [k]))
        self.decomposition = pd.concat(decomposition, axis=1).set_axis(self.input_df.index, axis=0)
        self.decomposition = self.scalers['y'].InverseTransform(self.decomposition)
        return self.decomposition
    
    def GetSamples(self):
        return self.model.mcmc.get_samples()
    
    def SitesNames(self):
        return self.model.mcmc.get_samples().keys()
    
    


























    ########################## *********CHARTS *****************

    def PlotInputs(self): 
        tiles_num = sum([len(section) for _, section in self.X.items()]) + 1
        _, axs = plt.subplots(tiles_num, 1, figsize=(16, tiles_num * 2))
        if tiles_num == 1:
            axs.plot(self.y, label='Dependent')
            axs.legend()
        else:
            axs[0].plot(self.y, label='Dependent')
            axs[0].legend()

        next_tile = 1
        for var in self.X['media']:
            pd.DataFrame(var.GetData(), columns=var.VarColumns()).plot.area(ax=axs[next_tile], linewidth=0, stacked=False)
            next_tile = next_tile + 1
        for var in self.X['non-media']:
            pd.DataFrame(var.GetData(), columns=var.VarColumns()).plot.line(ax=axs[next_tile])
            next_tile = next_tile + 1
        plt.show()

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
    
    def __GetNonMediaSites(self, include_seasonality=True):
        result = [MODEL_BASE]
        if self.seasonality_period > 1 and include_seasonality:
            result.append(MODEL_SEASONAL)
        for s in self.spec['X']:
            if 'media' not in s:
                result.append(s)
        return result
    
    def __GetOwnMediaSites(self):
        return [s for s in [MEDIA_OWN, MEDIA_OWN_LOW_RET] if s in self.spec['X']]

    def __GetCompMediaSites(self):
        return [s for s in [MEDIA_COMP] if s in self.spec['X']]
    
    def __GetMediaSites(self):
        return [s for s in [MEDIA_OWN, MEDIA_OWN_LOW_RET, MEDIA_COMP] if s in self.spec['X']]

    def PlotFit(self):
        if self.decomposition is None:
            self.GetDecomposition()
        
        fig, axs = plt.subplots(1, 1, figsize=(16, 6))

        self.decomposition[MODEL_Y].set_axis(['Modelled metric'], axis=1).plot.line(color='red', ax=axs)
        #pd.DataFrame(self.y, index=self.decomposition.index).plot.line(ax=axs, label="Observed metric", color='grey', alpha=0.5)
        self.input_df[self.spec[Y]].plot.line(ax=axs, label="Observed metric", color='grey', alpha=0.5)
        axs.legend()


    def PlotDecomposition(self, ylim: tuple=None):
        if self.decomposition is None:
            self.GetDecomposition()
        
        chart_data = pd.concat([
            self.decomposition[self.__GetNonMediaSites()].sum(1).rename("Non-media"), 
            self.decomposition[self.__GetOwnMediaSites()].sum(1).rename("Own media"),
            self.decomposition[self.__GetCompMediaSites()].sum(1).rename("Competitors media")
        ], axis=1)
        
        media_sites = self.__GetMediaSites()
        _, axs = plt.subplots(len(media_sites) + 1, 1, figsize=(16, 6), height_ratios=[7] + [1] * len(media_sites))
        
        for tile, media in enumerate(media_sites): 
            self.input_df[self.spec['X'][media]].sum(1).rename(media).plot.area(ax=axs[tile+1], linewidth=0, stacked=False)
            axs[tile+1].legend()
        
        AreaChartWithNegative(chart_data, axs[0], ylim=ylim)
        

    def PlotCompMediaDecomposition(self, ylim: tuple=None):
        if self.decomposition is None:
            self.GetDecomposition()
        
        _, axs = plt.subplots(2, 1, figsize=(16, 6), height_ratios=[5, 1])
        self.input_df[self.spec['X'][MEDIA_COMP]].plot.area(ax=axs[1], linewidth=0, stacked=False)
        AreaChartWithNegative(self.decomposition[MEDIA_COMP], axs[0], ylim=ylim)
        

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
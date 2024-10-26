import io

import pandas as pd
import numpy as np 
import numpyro

from smoother import Smoother
from sales_model import SalesModel

import matplotlib.pyplot as plt
#import seaborn as sns

#from utils import FixDirPath

from definitions import *


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
        print(self.min_, self.max_, self.scaling, self.centering)

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

    def Transform(self, data):
        #assert data.shape == self.fit_shape_
        return data.sub(self.min_).div(self.max_ - self.min_).sub(self.centering_shift)
    
    def InverseTransform(self, data):
        #assert data.shape == self.fit_shape_
        return data.add(self.centering_shift).mul(self.max_ - self.min_).add(self.min_)
    



class Modeller:
    """
    seasonality_period: int=1 //1 mean no seasonality used
    seasonality_model in ['discrete', 'fourier']
    seasonality_num_fouries_terms: int=None   // если не специфицировано будет seasonality_period / 4
    """
    seasonality_period = 1
    seasonality_model: str
    seasonality_num_fouries_terms: int = None
    model_name: str
    model_type: str

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


    def __init__(self, 
                 model_name,
                 diminishing_return: bool=True,
                 media_retention_factor: int=3, 
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
        self.model_name = model_name

    def CheckFixSpec(self, spec):
        if 'y' in spec: 
            if isinstance(spec['y'], list):
                spec['y'] = spec['y'][0]
        return spec

    @staticmethod
    def PrepareMediaX(df: pd.DataFrame) -> np.array:
        SMOOTH_WINDOW = 5
        df = df.fillna(0).rolling(window=SMOOTH_WINDOW, min_periods=1, center=False).mean()
        return df.div(df.max(axis=None)).values

    def PrepareInputs(self): 
        self.X = dict()
        self.scalers = dict()

        df = self.input_df[self.spec['y']]
        self.scalers['y'] = Scaler(scaling='max_only').Fit(df)
        self.y = Smoother().Impute(self.scalers['y'].Transform(df))

        if BRAND in self.spec['X']: 
            df = self.input_df[self.spec['X'][BRAND]]
            self.scalers[BRAND] = Scaler(scaling='min_max', scaler_from='column', centering='first').Fit(df)
            self.X[BRAND] = Smoother().Impute(self.scalers[BRAND].Transform(df))

        if PRICE in self.spec['X']: 
            df = self.input_df[self.spec['X'][PRICE]]
            self.scalers[PRICE] = Scaler(scaling='min_max', scaler_from='total', centering='first').Fit(df)
            self.X[PRICE] = Smoother().Impute(self.scalers[PRICE].Transform(df))

        if WSD in self.spec['X']: 
            df = self.input_df[self.spec['X'][WSD]]
            self.scalers[WSD] = Scaler(scaling='min_max', scaler_from='column', centering='first').Fit(df)
            self.X[WSD] = Smoother().Impute(self.scalers[WSD].Transform(df))
        
        if STRUCT in self.spec['X']: 
            df = self.input_df[self.spec['X'][STRUCT]]
            self.scalers[STRUCT] = Scaler(scaling='min_max', scaler_from='column', centering='first').Fit(df)
            self.X[STRUCT] = Smoother().Impute(self.scalers[STRUCT].Transform(df))

        if MEDIA_OWN in self.spec['X']:
            self.X[MEDIA_OWN] = Modeller.PrepareMediaX(self.input_df[self.spec['X'][MEDIA_OWN]])
        
        if MEDIA_COMP in self.spec['X']:
            self.X[MEDIA_COMP] = Modeller.PrepareMediaX(self.input_df[self.spec['X'][MEDIA_COMP]])

        return True
    
    def PrepNoFit(self, data: pd.DataFrame, spec: dict, show_progress=True):
        # to check inputs
        self.input_df = data
        self.spec = self.CheckFixSpec(spec)
        self.PrepareInputs()
        return self 
    
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
        
        decomposition = {}
        for k, v in sample.items():
            if k == MODEL_MEDIA:
                all_media = pd.DataFrame(v.mean(axis=0))
                if MEDIA_OWN in self.spec['X']:
                    names_ = self.spec['X'][MEDIA_OWN]
                    decomposition[MEDIA_OWN] = all_media.iloc[:, :len(names_)].set_axis(names_, axis=1)
                if MEDIA_COMP in self.spec['X']:
                    names_ = self.spec['X'][MEDIA_COMP]
                    decomposition[MEDIA_COMP] = all_media.iloc[:, -len(names_):].set_axis(names_, axis=1)
            else:
                decomposition[k] = pd.DataFrame(
                    v.mean(axis=0), 
                    columns=(self.spec['X'][k] if k in self.spec['X'] else None))
        self.decomposition = pd.concat(decomposition, axis=1).set_axis(self.input_df.index, axis=0)
        #self.decomposition[(SALES, 0)] = self.scalers[SALES].InverseTransform(self.decomposition[(SALES, 0)])
        return self.decomposition
    
    def GetSamples(self):
        return self.model.mcmc.get_samples()
    
    def SitesNames(self):
        return self.model.mcmc.get_samples().keys()
    
    





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

    def SetChartsSpec(self):
        # только base level
        self.base_sites = [MODEL_BASE]

        # все кроме base and media
        self.struct_sites = []
        if self.seasonality_period > 1:
            self.struct_sites.append(MODEL_SEASONAL)
        for s in self.spec['X']:
            if 'media' not in s:
                self.struct_sites.append(s)
    



    def PlotFit(self):
        if self.decomposition is None:
            self.GetDecomposition()
        
        fig, axs = plt.subplots(1, 1, figsize=(16, 6))

        self.decomposition[MODEL_Y].set_axis(['Modelled metric'], axis=1).plot.line(color='red', ax=axs)
        pd.DataFrame(self.y, index=self.decomposition.index).plot.line(ax=axs, label="Observed metric", color='grey', alpha=0.5)
        axs.legend()


    def PlotDecomposition(self):
        if self.decomposition is None:
            self.GetDecomposition()
        self.SetChartsSpec()
        fig, axs = plt.subplots(3, 1, figsize=(16, 6), height_ratios=[7, 1, 1])

        own_area_chart = pd.DataFrame(self.decomposition[self.base_sites + self.struct_sites].sum(axis=1), columns=['Non-media'])
        if MEDIA_OWN in self.X:
            own_area_chart[MEDIA_OWN] = self.decomposition[MEDIA_OWN].sum(axis=1)
            self.input_df[self.spec['X'][MEDIA_OWN]].plot.area(ax=axs[1], linewidth=0, stacked=False)
        own_area_chart.plot.area(ax=axs[0], linewidth=0)

        if MEDIA_COMP in self.X:
            pd.DataFrame(self.decomposition[MEDIA_COMP].sum(axis=1), columns=['Competitors media'])\
                .plot.area(ax=axs[0], linewidth=0, 
                           ylim=(self.decomposition[MEDIA_COMP].min(axis=None), 1))
            self.input_df[self.spec['X'][MEDIA_COMP]].plot.area(ax=axs[2], linewidth=0, stacked=False)
        

    def PlotNonmediaDecomposition(self):
        if self.decomposition is None:
            self.GetDecomposition()
        self.SetChartsSpec()
        fig, axs = plt.subplots(1, 1, figsize=(16, 6))
        
        # all struct + seasonal as lines 
        if self.struct_sites:
            self.decomposition[self.struct_sites].plot.line(ax=axs, label='Base level')
        # base as area 
        self.decomposition[self.base_sites].plot.area(ax=axs, linewidth=0, alpha=0.2)
        axs.set_ylim(-0.4, 1.2)
        axs.legend()

        
        
    def PlotMediaDecomposition(self):
        if not MEDIA_OWN in self.X:
            return 
        fig, axs = plt.subplots(3, 1, figsize=(16, 6), height_ratios=[7, 1, 1])
        
        self.decomposition[MEDIA_OWN].plot.area(ax=axs[0], linewidth=0, stacked=False)
        self.input_df[self.spec['X'][MEDIA_OWN]].plot.area(ax=axs[1], linewidth=0, stacked=False)
        if MEDIA_COMP in self.X:
            self.input_df[self.spec['X'][MEDIA_COMP]].plot.area(ax=axs[2], linewidth=0, stacked=False)


    """def PlotInitialData(self, chart=True): 
        fig, axs = plt.subplots(len(self.spec), 1, figsize=(15, len(self.spec)*3))

        self.input_df[self.spec[SALES]].plot.line(ax=axs[0], label=SALES)

        i = 1
        if BRAND in self.spec: 
            self.input_df[self.spec[BRAND]].plot.line(ax=axs[i], label=BRAND)
            axs[i].legend()
            i = i + 1
        if PRICE in self.spec: 
            self.input_df[self.spec[PRICE]].plot.line(ax=axs[i], label=PRICE)
            axs[i].legend()
            i = i + 1
        if WSD in self.spec: 
            self.input_df[self.spec[WSD]].plot.line(ax=axs[i], label=WSD)
            axs[i].legend()
            i = i + 1
        if MEDIA_OWN in self.spec:
            self.input_df[self.spec[MEDIA_OWN]].plot.area(ax=axs[i], linewidth=0, stacked=False)"""
        

    def PlotInputs(self): 
        tiles_num = len(self.X) + 1
        fig, axs = plt.subplots(tiles_num, 1, figsize=(12, tiles_num * 2))
        if tiles_num == 1:
            axs.plot(self.y, label='Dependent')
            axs.legend()
        else:
            axs[0].plot(self.y, label='Dependent')
            axs[0].legend()

        next_tile = 1
        for sec, d in self.X.items():
            if 'media' in sec:
                pd.DataFrame(d).plot.area(ax=axs[next_tile], linewidth=0, stacked=False)
            else:
                axs[next_tile].plot(d, label=sec)
                axs[next_tile].legend()
            next_tile = next_tile + 1
        plt.show()







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
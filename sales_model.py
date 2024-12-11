from math import ceil

import jax
import jax.numpy as jnp
import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan

from definitions import *

numpyro.set_host_device_count(4)


def FourierCovs(period: int, num_terms: int) -> np.array:
    fourier_terms = []
    #fourier_names = [] #for pandas
    time_axis = np.arange(period)
    for term in range(1, num_terms + 1):
        fourier_terms.append(np.sin(2 * term * np.pi * time_axis / period))
        #fourier_names.append("sin{}".format(2 * term))
        fourier_terms.append(np.cos(2 * term * np.pi * time_axis / period))
        #fourier_names.append("cos{}".format(2 * term))
    return np.column_stack(fourier_terms)

class SalesModel:
    # data
    data_len: int #!!!!!!!!!!!!! избавиться от этого! 
    mcmc = None 
    sample_model = None

    # seasonality
    seasonality: dict = None
    SeasonalityCovs: np.array = None
    
    def __init__(self, seasonality_spec: dict=None):
        self.seasonality = seasonality_spec
        if self.seasonality and (self.seasonality["model"] == 'fourier'):
            self.SeasonalityCovs = FourierCovs(self.seasonality["cycle"], self.seasonality["num_fouries_terms"])
    
    def __Model(self, X: dict, y=None):
        time_axis = jnp.arange(self.data_len) #нумерация данных - нужна для сезонности

        base_init =   numpyro.sample("base_init", dist.Beta(2, 2))
        base_drift_scale = numpyro.sample("base_drift_scale", dist.HalfNormal(0.005))
        noise_scale = numpyro.sample("noise_scale", dist.HalfCauchy(1))

        # seasonality variables
        if self.seasonality:
            if self.seasonality["model"] == 'fourier':
                _dims = self.SeasonalityCovs.shape[-1]
                seasonal = numpyro.deterministic("seasonal",
                    (
                        self.SeasonalityCovs\
                        * numpyro.sample('seasonality_betas', dist.Normal(0, 0.1).expand([_dims]).to_event(1))
                    ).sum(axis=1, keepdims=True)
                )
            elif self.seasonality["model"] == 'discrete':
                seasonal = numpyro.deterministic("seasonal",
                    numpyro.sample('seasonality_one_cycle', dist.Normal(0, 1).expand([self.seasonality_period]).to_event(1)), 
                )
        
        # media variables
        media_covs = []
        media_retentions = []
        for vg in X["media"]:
            _data = vg.GetData()
            if vg.Saturation():
                alpha = numpyro.sample("alpha {}".format(vg.Name()), dist.LogNormal(1, 1).expand([vg.Dims()]).to_event(1))
                gamma = numpyro.sample("gamma {}".format(vg.Name()), dist.Beta(1, 1).expand([vg.Dims()]).to_event(1))
                data_pow_alpha = jnp.power(_data, alpha)
                _data = data_pow_alpha / (data_pow_alpha + jnp.power(gamma, alpha))
            betas = numpyro.sample("raw beta {}".format(vg.Name()), dist.Normal(loc=0, scale=.05).expand([vg.Dims()]).to_event(1))
            betas = numpyro.deterministic("beta {}".format(vg.Name()), 
                                          vg.BetaVector() * jnp.where(vg.ForceVector(), jnp.abs(betas), betas)
                                          )
            media_covs.append(_data * betas)
            media_retentions.append(
                numpyro.sample("retention {}".format(vg.Name()), dist.Beta(*vg.RetentionVector()).to_event(1))
            )
        if len(media_covs) == 0:
            media_covs.append(jnp.zeros((self.data_len, 1)))
            media_retentions.append(jnp.zeros((1)))
        # вливаем все в одно медиа
        media_impact = jnp.column_stack(media_covs)
        retention = jnp.concatenate(media_retentions)

        # non-media variables
        non_media_covs = []
        non_media_betas = []
        for vg in X["non-media"]: 
            raw_betas = numpyro.sample("raw beta {}".format(vg.Name()), dist.Normal(1).expand([vg.Dims()]).to_event(1))
            non_media_betas.append(
                numpyro.deterministic("beta {}".format(vg.Name()), 
                                      vg.BetaVector() * jnp.where(vg.ForceVector(), jnp.abs(raw_betas), raw_betas)
                                      )
                                    )
            non_media_covs.append(vg.GetData())
        if len(non_media_covs) == 0:
            non_media_covs.append(jnp.zeros((self.data_len, 1)))
            non_media_betas.append(jnp.zeros(1))
        non_media_impact = (
            numpyro.deterministic("non-media", jnp.column_stack(non_media_covs) * jnp.concatenate(non_media_betas))
        ).sum(axis=1)
        
        def transition(carry, current):
            base_prev, media_prev = carry
            y_curr, media_curr, struct_curr, time_curr = current

            media_curr = numpyro.deterministic("media", retention * media_prev + media_curr)
            base_curr = numpyro.sample("base", dist.Normal(base_prev, base_drift_scale))
            seasonality_curr = seasonal[time_curr % self.seasonality["cycle"]] if self.seasonality else 0
            y_curr = numpyro.sample("y", dist.StudentT(2,
                base_curr + struct_curr + seasonality_curr + media_curr.sum(), 
                noise_scale), 
                obs=y_curr)
            
            return (base_curr, media_curr), y_curr 

        _, y = scan(transition, (base_init, jnp.zeros_like(retention)), (y, media_impact, non_media_impact, time_axis))
        return y

    def Fit(self, X: dict, y: np.array, show_progress=True, num_samples=1000):
        self.data_len = len(y)
        self.mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(self.__Model),
            num_warmup=1000,
            num_samples=num_samples,
            num_chains=4,
            progress_bar=show_progress,
        )
        rng_key = jax.random.PRNGKey(3)
        self.mcmc.run(rng_key, X=X, y=y)
        return self
    
    
    def SampleModel(self, X): 
        #сэмплирует модель. ключи аутпута: []
        assert self.mcmc, "Run .Fit first"
        pred_func = numpyro.infer.Predictive(self.__Model, 
                                             posterior_samples=self.mcmc.get_samples(), 
                                             return_sites=['y', 'base', 'media', 'non-media']
                                             )
        self.sample_model = pred_func(jax.random.PRNGKey(3), X)
        return self.sample_model
    

    def GetPredictionAndDecomposition(self, X, mode='mean', prob=0.5):
        # mode options ['mean']
        assert mode in ['hpdi', 'quantile', 'mean']
        if not self.sample_model: 
            self._SampleModel(X)
        if mode == 'quantile':
            raise NotImplementedError()
        if mode == 'hpdi':
            raise NotImplementedError()
            decompositions_mid_hpdi = numpyro.diagnostics.hpdi(self._SampleDecompositions(X), prob=prob).mean(axis=0) # shape = (time * [base, ads, y])
            decompositions_mid_hpdi[..., 0] = decompositions_mid_hpdi[..., -1] - decompositions_mid_hpdi[..., 1:-1].sum(axis=1)
            return decompositions_mid_hpdi[..., -1],  decompositions_mid_hpdi[..., :-1]
        #if mode == 'mean':

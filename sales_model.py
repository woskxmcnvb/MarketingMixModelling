from math import ceil

import jax
import jax.numpy as jnp
import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan

#from .modeller import ModelInputs

from .definitions import *

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

    # options
    fixed_base: bool = False
    long_term_retention: int | tuple = 1
    
    def __init__(self, seasonality_spec: dict=None, fixed_base: bool=False, long_term_retention: int | tuple = 1):
        self.seasonality = seasonality_spec.copy()
        self.fixed_base = fixed_base
        self.long_term_retention = long_term_retention
        if self.seasonality and (self.seasonality["model"] == 'fourier'):
            self.SeasonalityCovs = FourierCovs(self.seasonality["cycle"], self.seasonality["num_fouries_terms"])
    
    def Model(self, inputs: ModelInputs, y=None):
        time_axis = jnp.arange(self.data_len) #нумерация данных - нужна для сезонности

        base_init =   numpyro.sample("base_init", dist.Beta(2, 2))
        
        if not self.fixed_base:
            base_drift_scale = numpyro.sample("base_drift_scale", dist.HalfNormal(0.05))
        noise_scale = numpyro.sample("noise_scale", dist.HalfCauchy(1))
        
        if self.long_term_retention == 1: 
            retention_long = numpyro.deterministic("retention long", jnp.array(1))
        else: 
            retention_long = numpyro.sample("retention long", dist.Beta(100, 1))
        
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
        
        if len(inputs.media_vars) == 0:
            media_short_covs = jnp.zeros((self.data_len, 1))
            retention_short = jnp.zeros((1))
            media_long_covs = jnp.zeros((self.data_len, 1))
        else: 
            alpha = []
            gamma = []
            beta_short = []
            beta_long = []
            retention_short = []
            for vg in inputs.media_vars:
                # saturation
                if vg.Saturation() == 'local':
                    alpha.append(
                        numpyro.sample("alpha {}".format(vg.Name()), dist.LogNormal(1, 1).expand([vg.Dims()]).to_event(1))
                    )
                    gamma.append(
                        numpyro.sample("gamma {}".format(vg.Name()), dist.Beta(1, 1).expand([vg.Dims()]).to_event(1))
                    )
                
                elif vg.Saturation() == 'global':
                    alpha.append(
                        jnp.tile(numpyro.sample("alpha {}".format(vg.Name()), dist.LogNormal(1, 1)), vg.Dims())
                    )
                    gamma.append(
                        jnp.tile(numpyro.sample("gamma {}".format(vg.Name()), dist.Beta(1, 1)), vg.Dims())
                    )
                
                # short-term retention
                if vg.GlobalRetention():
                    retention_short.append(
                        numpyro.deterministic("retention {}".format(vg.Name()),
                            jnp.tile(numpyro.sample("raw retention {}".format(vg.Name()), dist.Beta(*vg.GlobalRetention())), vg.Dims())
                        )
                    )
                else: 
                    retention_short.append(
                        numpyro.sample("retention {}".format(vg.Name()), dist.Beta(*vg.RetentionVector()).to_event(1))
                    )
                
                # short-term betas
                _beta = numpyro.sample("raw beta {}".format(vg.Name()), dist.Normal(loc=0, scale=.05).expand([vg.Dims()]).to_event(1))
                beta_short.append(
                    numpyro.deterministic("beta {}".format(vg.Name()), 
                                        vg.BetaVector() * jnp.where(vg.ForceVector(), jnp.abs(_beta), _beta))
                )

                # long-term betas and covs
                if vg.HasLongTermEffect():
                    _beta = numpyro.sample("raw beta long {}".format(vg.Name()), dist.Normal(loc=0, scale=.05).expand([vg.Dims()]).to_event(1))
                    beta_long.append(
                        numpyro.deterministic("beta long {}".format(vg.Name()), 
                            vg.BetaVector() * jnp.where(vg.ForceVector(), jnp.abs(_beta), _beta))
                    )
                else:
                    beta_long.append(
                        numpyro.deterministic("beta long {}".format(vg.Name()), jnp.zeros(vg.Dims()))
                    )
        
            # предобработка медиа
            alpha = jnp.concatenate(alpha)
            gamma = jnp.concatenate(gamma)
            beta_short = jnp.concatenate(beta_short)
            beta_long = jnp.concatenate(beta_long)
            retention_short = jnp.concatenate(retention_short)
        
            data_pow_alpha = jnp.power(inputs.media_data, alpha)
            media_short_covs = beta_short * (data_pow_alpha / (data_pow_alpha + jnp.power(gamma, alpha)))
            media_long_covs = beta_long * inputs.media_data
        

        # non-media variables
        if len(inputs.non_media_vars) == 0:
            non_media_covs = numpyro.deterministic("non-media", jnp.zeros((self.data_len, 1)))
        else: 
            non_media_betas = []
            for vg in inputs.non_media_vars: 
                raw_betas = numpyro.sample("raw beta {}".format(vg.Name()), dist.Normal(1).expand([vg.Dims()]).to_event(1))
                non_media_betas.append(
                    numpyro.deterministic("beta {}".format(vg.Name()), 
                                        vg.BetaVector() * jnp.where(vg.ForceVector(), jnp.abs(raw_betas), raw_betas))
                )
            non_media_covs = numpyro.deterministic("non-media", 
                jnp.concatenate(non_media_betas) * inputs.non_media_data
            )
        # cливаем все в один СТОЛБЕЦ 
        non_media_impact = non_media_covs.sum(axis=1)
        
        # cливаем все медиа в один СТОЛБЕЦ 

        
        def transition(carry, current):
            base_prev, media_short_prev, media_long_prev = carry
            y_curr, media_short_curr, media_long_curr, struct_curr, time_curr = current

            media_curr = numpyro.deterministic("media short", retention_short * media_short_prev + media_short_curr)
            media_long_curr = numpyro.deterministic("media long", retention_long * media_long_prev + media_long_curr)
            if self.fixed_base:
                base_curr = numpyro.deterministic("base", base_prev)
            else:
                base_curr = numpyro.sample("base", dist.Normal(base_prev, base_drift_scale))
            seasonality_curr = seasonal[time_curr % self.seasonality["cycle"]] if self.seasonality else 0
            y_curr = numpyro.sample("y", dist.StudentT(2,
                base_curr + struct_curr + seasonality_curr + media_curr.sum() + media_long_curr.sum(), 
                noise_scale), 
                obs=y_curr)
            
            return (base_curr, media_curr, media_long_curr), y_curr 

        _, y = scan(transition, 
                    (base_init, jnp.zeros(media_short_covs.shape[-1]), jnp.zeros(media_long_covs.shape[-1])),
                    (y, media_short_covs, media_long_covs, non_media_impact, time_axis))
        return y

    def Fit(self, model_inputs: ModelInputs, show_progress=True, num_samples=1000):
        self.data_len = model_inputs.y.shape[0]
        self.mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(self.Model),
            num_warmup=1000,
            num_samples=num_samples,
            num_chains=4,
            progress_bar=show_progress,
        )
        rng_key = jax.random.PRNGKey(3)
        self.mcmc.run(rng_key, model_inputs, y=model_inputs.y)
        return self
    
    
    def Predict(self, X): 
        assert self.mcmc, "Run .Fit first"
        pred_func = numpyro.infer.Predictive(self.Model, 
                                             posterior_samples=self.mcmc.get_samples(), 
                                             return_sites=['y', 'base', 'media short', 'media long', 'non-media']
                                             )
        self.sample_model = pred_func(jax.random.PRNGKey(3), X)
        return self.sample_model
    

    """def GetPredictionAndDecomposition(self, X, mode='mean', prob=0.5):
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
        #if mode == 'mean':"""

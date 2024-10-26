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
    data_len: int #!!!!!!!!!!!!! избавиться от этого! 
    mcmc = None 
    sample_model = None
    non_cov_sites: list = [MODEL_Y, MODEL_BASE, MODEL_MEDIA]
    
    # model settings
    retention_factor: int
    diminishing_return: bool

    # seasonality
    seasonality_period: int = 1
    seasonality_model: str
    SeasonalityCovs: np.array = None
    
    def __init__(self, 
                 diminishing_return: bool=True,
                 seasonality_period: int=1, 
                 seasonality_model: str='discrete',
                 seasonality_num_fouries_terms: int=None, 
                 retention_factor: int=3):
        self.diminishing_return = diminishing_return
        self.retention_factor = retention_factor
        self.seasonality_period = seasonality_period
        self.seasonality_model = seasonality_model
        if self.seasonality_period > 1: 
            self.non_cov_sites = self.non_cov_sites + [MODEL_SEASONAL]
            if self.seasonality_model == 'fourier':
                self.SeasonalityCovs = FourierCovs(self.seasonality_period, seasonality_num_fouries_terms)
    
    def __Model(self, X: dict, y=None):
        time_axis = jnp.arange(self.data_len) #нумерация данных - нужна для сезонности

        base_init =   numpyro.sample("base_init", dist.Beta(2, 2))
        base_drift_scale = numpyro.sample("base_drift_scale", dist.HalfNormal(1))
        noise_scale = numpyro.sample("noise_scale", dist.HalfCauchy(1))

        if self.seasonality_period > 1:
            if self.seasonality_model == 'fourier':
                _dims = self.SeasonalityCovs.shape[-1]
                seasonal = numpyro.deterministic(MODEL_SEASONAL,
                    (
                        self.SeasonalityCovs\
                        * numpyro.sample('seasonality_betas', dist.Normal(0, 0.1).expand([_dims]).to_event(1))
                    ).sum(axis=1, keepdims=True)
                )
            elif self.seasonality_model == 'discrete':
                seasonal = numpyro.deterministic(MODEL_SEASONAL,
                    numpyro.sample('seasonality_one_cycle', dist.Normal(0, 1).expand([self.seasonality_period]).to_event(1)), 
                )
            else:
                raise ValueError("some shit with seasonality spec")
        else:
            seasonal = jnp.zeros((self.data_len, 1))
        
        media_covs = []
        media_retentions = []
        if MEDIA_OWN in X.keys():
            _dims = X[MEDIA_OWN].shape[-1]
            if self.diminishing_return:
                alpha = numpyro.sample("alpha", dist.LogNormal(1, 1).expand([_dims]).to_event(1))
                gamma = numpyro.sample("gamma", dist.Beta(1,1).expand([_dims]).to_event(1))
                x_pow_alpha = jnp.power(X[MEDIA_OWN], alpha)
                X_media = x_pow_alpha / (x_pow_alpha + jnp.power(gamma, alpha))
            else: 
                X_media = X[MEDIA_OWN]
            media_covs.append(
                X_media * numpyro.sample("media_beta", dist.HalfNormal(.05).expand([_dims]).to_event(1)))
            media_retentions.append(
                numpyro.sample("media_retention", dist.Beta(3, 1).expand([_dims]).to_event(1))
            )
        if MEDIA_COMP in X.keys():
            _dims = X[MEDIA_COMP].shape[-1]
            media_covs.append(
                - X[MEDIA_COMP] * numpyro.sample("comp_media_beta", dist.HalfNormal(.05).expand([_dims]).to_event(1)))
            media_retentions.append(
                numpyro.sample("comp_media_retention", dist.Beta(3, 1).expand([_dims]).to_event(1))
            )
        if len(media_covs) == 0:
            media_covs.append(jnp.zeros((self.data_len, 1)))
            media_retentions.append(jnp.zeros((1)))
        # вливаем все в одно медиа
        media_impact = jnp.column_stack(media_covs)
        retention = jnp.concatenate(media_retentions)

        struct_covs = []
        if PRICE in X.keys():
            price_beta = numpyro.sample("price_beta", dist.Normal(1).expand([X[PRICE].shape[-1]]).to_event(1))
            struct_covs.append(
                numpyro.deterministic(PRICE, X[PRICE] * price_beta).sum(axis=1, keepdims=True)
            )
        if WSD in X.keys():
            wsd_beta = numpyro.sample("wsd_beta", dist.HalfNormal(1).expand([X[WSD].shape[-1]]).to_event(1))
            struct_covs.append(
                numpyro.deterministic(WSD, X[WSD] * wsd_beta).sum(axis=1, keepdims=True)
            )
        if STRUCT in X.keys():
            struct_beta = numpyro.sample("structural_beta", dist.Normal(1).expand([X[STRUCT].shape[-1]]).to_event(1))
            struct_covs.append(
                numpyro.deterministic(STRUCT, X[STRUCT] * struct_beta).sum(axis=1, keepdims=True)
            )
        if BRAND in X.keys():
            brand_beta = numpyro.sample("brand_beta", dist.HalfNormal(1).expand([X[BRAND].shape[-1]]).to_event(1))
            struct_covs.append(
                numpyro.deterministic(BRAND, X[BRAND] * brand_beta).sum(axis=1, keepdims=True)
            )
        if len(struct_covs) == 0:
            struct_covs.append(jnp.zeros((self.data_len, 1)))
        structural = jnp.column_stack(struct_covs).sum(axis=1)
        
        def transition(carry, current):
            base_prev, media_prev = carry
            y_curr, media_curr, struct_curr, time_curr = current

            media_curr = numpyro.deterministic(MODEL_MEDIA, retention * media_prev + media_curr)
            base_curr = numpyro.sample(MODEL_BASE, dist.Normal(base_prev, base_drift_scale))
            y_curr = numpyro.sample(MODEL_Y, dist.StudentT(2,
                base_curr + struct_curr + seasonal[time_curr % self.seasonality_period] + media_curr.sum(), 
                noise_scale), 
                obs=y_curr)
            
            return (base_curr, media_curr), y_curr 

        _, y = scan(transition, (base_init, jnp.zeros_like(retention)), (y, media_impact, structural, time_axis))
        return y


    def Fit(self, X: dict, y: np.array, show_progress=True, num_samples=1000):
        self.data_len = len(y)
        self.num_seasons_in_data = ceil(self.data_len / self.seasonality_period)
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
                                             return_sites=self.non_cov_sites + list(X.keys()))
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

    
    
             
   
    

"""def __Model(self, X: dict, y=None):
        data_len=len(next(iter(X.values())))

        sales_init_base =   numpyro.sample("base_init", dist.Beta(2, 2))
        sales_drift_scale = numpyro.sample("base_drift_scale", dist.HalfNormal(1))
        sales_noise_scale = numpyro.sample("sales_noise_scale", dist.HalfCauchy(1))
        
        structural = jnp.zeros((data_len, 1))
        media_impact = jnp.zeros((data_len, 1))
        
        if 'media' in X.keys():
            media_dim = X['media'].shape[-1]
            retention =  numpyro.sample("media_retention", dist.Beta(5, 1).expand([media_dim]).to_event(1))
            media_beta = numpyro.sample("media_beta", dist.HalfNormal(1).expand([media_dim]).to_event(1))
            media_impact = X['media'] * media_beta

        if 'price' in X.keys():
            price_dim = X['price'].shape[-1]
            price_beta = numpyro.sample("price_beta", dist.HalfNormal(1).expand([price_dim]).to_event(1))
            price = numpyro.deterministic("price", - X['price'] * price_beta)
            structural = structural + price.sum(axis=1)
  
        if 'wsd' in X.keys():
            wsd_beta = numpyro.sample("wsd_beta", dist.HalfNormal(1).expand([1]).to_event(1))
            wsd = numpyro.deterministic("wsd", X['wsd'] * wsd_beta)
            structural = structural + wsd
        
        def transition(carry, current):
            b_prev, s_prev = carry
            y_curr, media_curr, struct_curr = current

            s_curr = numpyro.deterministic("media", retention * s_prev + media_curr)

            b_curr = numpyro.sample("base", dist.Normal(b_prev, sales_drift_scale))
            y_curr = numpyro.sample("y", dist.Normal(b_curr + struct_curr + s_curr.sum(), sales_noise_scale), obs=y_curr)
            
            return (b_curr, s_curr), y_curr

        _, y = scan(transition, (sales_init_base, jnp.zeros_like(retention)), (y, media_impact, structural))
        return y"""
    
"""def __Model_seasonal(self, X: dict, y=None):
        data_len=len(next(iter(X.values())))
        time_axis = jnp.arange(data_len) #нумерация данных - нужна для сезонности

        sales_init_base =   numpyro.sample("base_init", dist.Beta(2, 2))
        sales_drift_scale = numpyro.sample("base_drift_scale", dist.HalfNormal(1))
        sales_noise_scale = numpyro.sample("sales_noise_scale", dist.HalfCauchy(1))

        seasonal = numpyro.sample("seasonal", dist.Normal(0, 1).expand([self.seasonality]).to_event(1))
        
        structural = jnp.zeros((data_len, 1))
        media_impact = jnp.zeros((data_len, 1))
        
        if 'media' in X.keys():
            media_dim = X['media'].shape[-1]
            retention =  numpyro.sample("media_retention", dist.Beta(5, 1).expand([media_dim]).to_event(1))
            media_beta = numpyro.sample("media_beta", dist.HalfNormal(1).expand([media_dim]).to_event(1))
            media_impact = X['media'] * media_beta

        if 'price' in X.keys():
            price_dim = X['price'].shape[-1]
            price_beta = numpyro.sample("price_beta", dist.HalfNormal(1).expand([price_dim]).to_event(1))
            price = numpyro.deterministic("price", - X['price'] * price_beta)
            structural = structural + price.sum(axis=1, keepdims=True)
  
        if 'wsd' in X.keys():
            wsd_beta = numpyro.sample("wsd_beta", dist.HalfNormal(1).expand([1]).to_event(1))
            wsd = numpyro.deterministic("wsd", X['wsd'] * wsd_beta)
            structural = structural + wsd
        
        def transition(carry, current):
            b_prev, s_prev = carry
            y_curr, media_curr, struct_curr, time_curr = current

            s_curr = numpyro.deterministic("media", retention * s_prev + media_curr)

            b_curr = numpyro.sample("base", dist.Normal(b_prev, sales_drift_scale))
            y_curr = numpyro.sample("y", dist.Normal(
                b_curr + struct_curr + seasonal[time_curr % self.seasonality] + s_curr.sum(), 
                sales_noise_scale), 
                obs=y_curr)
            
            return (b_curr, s_curr), y_curr

        _, y = scan(transition, (sales_init_base, jnp.zeros_like(retention)), (y, media_impact, structural, time_axis))
        return y"""
    


"""# попытка встроить бренд в единую модель - WORK
    def __Model_seasonal(self, X: dict, y=None):
        data_len=len(next(iter(X.values())))
        time_axis = jnp.arange(data_len) #нумерация данных - нужна для сезонности

        sales_init_base =   numpyro.sample("base_init", dist.Beta(2, 2))
        sales_drift_scale = numpyro.sample("base_drift_scale", dist.HalfNormal(1))
        sales_noise_scale = numpyro.sample("sales_noise_scale", dist.HalfCauchy(1))

        brand_init_base =   numpyro.sample("brand_base_init", dist.Beta(2, 2))
        brand_drift_scale = numpyro.sample("brand_base_drift_scale", dist.HalfNormal(1))
        brand_noise_scale = numpyro.sample("brand_noise_scale", dist.HalfCauchy(1))

        seasonal = numpyro.sample("seasonal", dist.Normal(0, 1).expand([self.seasonality]).to_event(1))
        
        structural = jnp.zeros((data_len, 1))
        media_impact = jnp.zeros((data_len, 1))
        
        if 'media' in X.keys():
            media_dim = X['media'].shape[-1]
            retention =  numpyro.sample("media_retention", dist.Beta(5, 1).expand([media_dim]).to_event(1))
            media_beta = numpyro.sample("media_beta", dist.HalfNormal(1).expand([media_dim]).to_event(1))
            media_impact = X['media'] * media_beta

        if 'price' in X.keys():
            price_dim = X['price'].shape[-1]
            price_beta = numpyro.sample("price_beta", dist.HalfNormal(1).expand([price_dim]).to_event(1))
            price = numpyro.deterministic("price", - X['price'] * price_beta)
            structural = structural + price.sum(axis=1, keepdims=True)
  
        if 'wsd' in X.keys():
            wsd_beta = numpyro.sample("wsd_beta", dist.HalfNormal(1).expand([1]).to_event(1))
            wsd = numpyro.deterministic("wsd", X['wsd'] * wsd_beta)
            structural = structural + wsd

        brand_beta = numpyro.sample("brand_beta", dist.HalfNormal(1))
        
        def transition(carry, current):
            b_prev, s_prev, brand_base_prev = carry
            y_curr, brand_curr, media_curr, struct_curr, time_curr = current

            s_curr = numpyro.deterministic("media", retention * s_prev + media_curr)

            brand_base_curr = numpyro.sample("brand_base", dist.Normal(brand_base_prev, brand_drift_scale))
            brand_curr = numpyro.sample("brand", dist.Normal(brand_base_curr, brand_noise_scale), obs=brand_curr)
            
            b_curr = numpyro.sample("base", dist.Normal(b_prev, sales_drift_scale))
            y_curr = numpyro.sample("y", dist.Normal(
                b_curr + struct_curr + seasonal[time_curr % self.seasonality] + s_curr.sum() + brand_beta * brand_base_curr, 
                sales_noise_scale), 
                obs=y_curr)
            
            return (b_curr, s_curr, brand_base_curr), (y_curr) #, brand_curr) 

        _, y = scan(transition, (sales_init_base, jnp.zeros_like(retention), brand_init_base), (y, X['brand'], media_impact, structural, time_axis))
        return y"""
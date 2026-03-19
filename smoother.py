import jax
import jax.numpy as jnp

import numpy as np
import pandas as pd

import numpyro
import numpyro.distributions as dist

from pykalman import KalmanFilter
from sklearn.preprocessing import MinMaxScaler

numpyro.set_host_device_count(4)

class Smoother:

    signal_to_noise_ratio = 0.1
    
    def Model(self, data_len, seasonality, y=None):
        init_base = numpyro.sample("init_base", dist.Beta(2, 2))
        # экспериментировал в Байере
        #drift_scale = numpyro.sample("drift_scale", dist.HalfNormal(0.01))
        drift_scale = numpyro.sample("drift_scale", dist.HalfNormal(self.signal_to_noise_ratio))
        noise_scale = numpyro.sample("noise_scale", dist.HalfCauchy(1))

        def transition(carry, current):
            b_curr = numpyro.sample("base", dist.Normal(carry, drift_scale))
            y_curr = numpyro.sample("y", dist.StudentT(2, b_curr, noise_scale), obs=current)
            return b_curr, y_curr

        _, y = numpyro.contrib.control_flow.scan(transition, init_base, y)    
        return y
    
    @staticmethod
    def Model_seasonal(data_len, seasonality, y=None):
        time_axis = jnp.arange(data_len) #нумерация данных - нужна для сезонности

        init_base = numpyro.sample("init_base", dist.Beta(2, 2))
        drift_scale = numpyro.sample("drift_scale", dist.HalfNormal(0.001))
        noise_scale = numpyro.sample("noise_scale", dist.HalfCauchy(1))

        seasonal = numpyro.sample("seasonal", dist.Normal(0, 1).expand([seasonality]).to_event(1))

        def transition(carry, current):
            y_curr, time_curr = current
            b_curr = numpyro.sample("base", dist.Normal(carry, drift_scale))
            y_curr = numpyro.sample("y", dist.StudentT(2, b_curr + seasonal[time_curr % seasonality], noise_scale), obs=y_curr)
            return b_curr, y_curr

        _, y = numpyro.contrib.control_flow.scan(transition, init_base, (y, time_axis))    
        return y
    
    def __FitMCMC(self, data, seasonality=0): 
        if seasonality: 
            model_to_use = Smoother.Model_seasonal
        else:
            model_to_use = self.Model
        
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(model_to_use), 
            num_warmup=1000, num_samples=1000, num_chains=4,
            progress_bar=True
        )
        rng_key = jax.random.PRNGKey(5)
        print("fitting smoother...")
        mcmc.run(rng_key, data_len=len(data), seasonality=seasonality, y=data)
        print("... done")
        return mcmc
    
    """def GetMCMCImputed(self):
        smoothed = self.GetMCMCSmoothed()
        return jnp.where(jnp.isnan(self.data), smoothed, self.data)"""
    
    @staticmethod
    def GetRolling(data, window=4):
        return np.pad(
            np.convolve(data, np.ones(window), mode='valid') / window, 
            (window-1, 0), 
            'constant', 
            constant_values = np.nan
        )
    
    def __FitKalman(self, data_na_masked: np.array): 
        return KalmanFilter(transition_matrices=[1], 
            observation_matrices=[1], 
            initial_state_mean = data_na_masked[:20].mean(),
            initial_state_covariance= data_na_masked[:20].var(),
            em_vars=['transition_covariance', 'observation_covariance']
            ).em(data_na_masked, n_iter=5)

    def __SmoothKalman(self, data: np.array) -> np.array:
        data_na_masked = np.ma.masked_array(data, np.isnan(data))
        kf = self.__FitKalman(data_na_masked)
        smoothed, _ = kf.smooth(data_na_masked)
        return smoothed.squeeze(-1)
    
    def __SmoothMCMC(self, data, seasonality=0):
        minimum, maximum = data.min(), data.max()
        mcmc = self.__FitMCMC((data - minimum) / (maximum - minimum), seasonality=seasonality)
        return mcmc.get_samples()['base'].mean(axis=0) * (maximum - minimum) + minimum
    
    def Smooth(self, data: np.array, method='kalman', signal_to_noise_ratio=0.1, seasonality=0) -> np.array: 
        # method = "kalman"
        assert method in ['kalman', 'MCMC']

        self.signal_to_noise_ratio = signal_to_noise_ratio

        if method == 'kalman':
            return self.__SmoothKalman(data)
        if method == 'MCMC':
            return self.__SmoothMCMC(data, seasonality=seasonality)
        
    def __Impute_array(self, data: np.array, method='kalman') -> np.array: 
        # method = "kalman"
        assert method in ['kalman'] #, 'MCMC']
        smoothed = self.Smooth(data, method=method)
        return np.where(np.isnan(data), smoothed, data)

    def Impute(self, data, method='kalman') -> np.array: 
        # method = "kalman"
        data = np.array(data)
        if not np.any(np.isnan(data)):
            return data
        if len(data.shape) == 1: 
            return self.__Impute_array(data)
        elif len(data.shape) == 2:
            return np.column_stack([self.__Impute_array(data[:, c]) for c in range(data.shape[-1])])



        
    
    
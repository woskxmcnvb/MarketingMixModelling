import scipy.optimize
import functools
import jax

from .definitions import *
from .modeller import Modeller
from .sales_model import SalesModel 


def _check_fix_opt_index(period: int | list, media_index: list, df: pd.DataFrame, modeller: Modeller) -> tuple:
    for i in media_index: 
        assert 0 <= i
        assert i <= modeller.X.media_data.shape[-1]
    
    if period is None: 
         return (Ellipsis, tuple(media_index))
    if isinstance(period, int): 
        assert period <= len(df)
        return (slice(len(df) - period, len(df)), tuple(media_index))
    if isinstance(period, list):
        assert len(period) == 2
        assert (0 <= period[0]) and (period[1] <= len(df))
        return (slice(period[0], period[1]), tuple(media_index))
    else: 
        raise ValueError("Wrong period_to_optimize")

def _check_spec_consistency(opt_index, X, keep_pattern): 
    TOLERANCE = 0.05
    if keep_pattern:
        channel_shares = X.media_data[opt_index].sum(axis=0) / X.media_data[opt_index].sum(axis=None)
        assert (channel_shares > TOLERANCE).all(),\
            "Not enough spends for the period to keep pattern. Check shares {}".format(channel_shares)


def _get_starting_values(opt_index: tuple, X) -> jnp.array: 
    sum_ = X.media_data[opt_index].sum(axis=0)
    return sum_ / sum_.sum()

def _constraints(alloc): 
    return jnp.sum(alloc) - 1

def _get_optimization_bounds(historical_values) -> scipy.optimize.Bounds:
    return scipy.optimize.Bounds(jnp.zeros_like(historical_values), jnp.ones_like(historical_values))

@jax.jit
def _redistribute_keep_pattern(media: jnp.array, allocation: jnp.array) -> jnp.array:
    return jnp.nan_to_num(media / media.sum(axis=0) * media.sum(axis=None) * allocation)

@jax.jit
def _redistribute_evenly(media: jnp.array, allocation: jnp.array) -> jnp.array:
    return jnp.ones_like(media) * media.sum(axis=None) * allocation / media.shape[0]

@functools.partial(
    jax.jit,
    static_argnames=("opt_index", "X", "modeller", "keep_spend_pattern"))
def _objective_function(
    media_allocation: jnp.array, 
    X: ModelCovs, 
    opt_index: tuple, 
    modeller: Modeller, 
    keep_spend_pattern: bool=True) -> jnp.array:
    
    if keep_spend_pattern:
        X.media_data = X.media_data.at[opt_index].set(_redistribute_keep_pattern(X.media_data[opt_index], media_allocation))
    else: 
        X.media_data = X.media_data.at[opt_index].set(_redistribute_evenly(X.media_data[opt_index], media_allocation))
    return -jnp.sum(jnp.mean(modeller.model.PredictY(X)['y'], axis=0))
   
def Optimize(df: pd.DataFrame, 
             media_to_optimize_index: list,
             period_to_optimize: int | list | None,
             modeller: Modeller, 
             keep_spend_pattern: bool = True):
    opt_index = _check_fix_opt_index(period_to_optimize, media_to_optimize_index, df, modeller)
    
    X = modeller.PrepareNewCovs(df)
    _check_spec_consistency(opt_index, X, keep_spend_pattern)

    # getting starting values as historical means, not from new data
    starting_values = _get_starting_values(opt_index, modeller.X)
    bounds = _get_optimization_bounds(starting_values)

    jax.config.update("jax_enable_x64", True)

    partial_objective_function = functools.partial(
        _objective_function, 
        X=X, opt_index=opt_index, modeller=modeller, keep_spend_pattern=keep_spend_pattern)

    opt_result = scipy.optimize.minimize(fun=partial_objective_function,
                                       x0=starting_values, 
                                       method="SLSQP",
                                       jac="3-point", 
                                       bounds=bounds,
                                       options={
                                          "maxiter": 200,
                                          "disp": True,
                                          "ftol": 1e-06,
                                          "eps": 1.4901161193847656e-08,
                                          },
                                      constraints={
                                          "type": "eq",
                                          "fun": _constraints
                                          }
                                      )
    return opt_result

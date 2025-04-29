import scipy.optimize
import functools
import jax

from .definitions import *
from .modeller import Modeller
from .sales_model import SalesModel 

def _check_fix_period(period: int | list | tuple, df: pd.DataFrame) -> slice:
    if (period is None) or (period is Ellipsis): 
         return Ellipsis
    if isinstance(period, int): 
        assert period <= len(df)
        return slice(len(df) - period, len(df))
    if isinstance(period, (tuple, list)):
        assert len(period) == 2
        assert (0 <= period[0]) and (period[1] <= len(df))
        return slice(period[0], period[1])
    else: 
        raise ValueError("Wrong period_to_optimize")

def _check_fix_media_vars(vars_: list, modeller: Modeller) -> tuple:
    for i in vars_: 
        assert 0 <= i
        assert i <= modeller.X.media_data.shape[-1]
    return tuple(vars_)

def _check_fix_non_media_vars(vars_: list, modeller: Modeller) -> tuple:
    for i in vars_: 
        assert 0 <= i
        assert i <= modeller.X.non_media_data.shape[-1]
    return tuple(vars_)



############# media optimization begin ############# 
def _check_media_optimize_spec(opt_index, X, keep_pattern): 
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
    opt_index = (
        _check_fix_period(period_to_optimize, df), 
        _check_fix_media_vars(media_to_optimize_index, modeller)
    )
    
    X = modeller.PrepareNewCovs(df)
    _check_media_optimize_spec(opt_index, X, keep_spend_pattern)

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


############# media optimization end ############# 

############# target reach begin ############# 
@jax.jit
def _inverse_scale(data, min_, max_, center_):
    return (data + center_) * (max_ - min_) + min_

@jax.jit
def _scale(data, min_, max_, center_):
    return (data - min_) / (max_ - min_) - center_

@functools.partial(
    jax.jit,
    static_argnames=("vars_to_adjust", "period_to_adjust", "target_value", "X", "modeller"))
def _target_function_non_media(
    multiplier: jnp.array,
    vars_to_adjust: list[int],
    period_to_adjust: slice,
    target_value: float,
    X: ModelCovs,
    starting_data: jnp.array,
    modeller: Modeller) -> jnp.array:
    
    X.non_media_data = _scale(
        starting_data.at[period_to_adjust, vars_to_adjust].set(
            starting_data[period_to_adjust, vars_to_adjust] * multiplier
        ), *X.NonMediaMinMaxCenter()
    )
    return jnp.abs(target_value - jnp.sum(jnp.mean(modeller.model.PredictY(X)['y'], axis=0)[period_to_adjust]))

def _target_function_media(
    multiplier: jnp.array,
    vars_to_adjust: list[int],
    period_to_adjust: slice,
    target_value: float,
    X: ModelCovs,
    starting_data: jnp.array,
    modeller: Modeller) -> jnp.array:
    
    X.media_data = starting_data.at[period_to_adjust, vars_to_adjust].set(
        starting_data[period_to_adjust, vars_to_adjust] * multiplier
    )
    return jnp.abs(target_value - jnp.sum(jnp.mean(modeller.model.PredictY(X)['y'], axis=0)[period_to_adjust]))

def ReachTarget(df: pd.DataFrame,
                relative_target: float,
                period_to_adjust: int | list | None,
                vars_to_adjust: list[int],
                vars_type: str,
                modeller: Modeller):
    """
    df: dataframe to work with
    relative_target: 1.1 means 10% increase vs. current etc.
    period_to_adjust: period to manipulate with and watch target. int - last x datapoints / list - from to / None - all 
    vars_to_adjust: vars to manipulate with 
    vars_type: 'media'/'non-media'
    modeller: fitted model
    """
    
    period_to_adjust = _check_fix_period(period_to_adjust, df)
    vars_to_adjust = _check_fix_non_media_vars(vars_to_adjust, modeller)
    
    XX = modeller.PrepareNewCovs(df)
    current_value = modeller.model.PredictY(XX)['y'].mean(axis=0)[period_to_adjust].sum()
    target_value = float(relative_target * current_value)
    starting_values = jnp.ones(len(vars_to_adjust))
    bounds = scipy.optimize.Bounds(starting_values * 0.5, starting_values * 1.5)
    print("Current value: {}, Target value: {}".format(current_value, target_value))
    print("Starting values: {}, bounds: {}".format(starting_values, bounds))

    if vars_type == 'media':
        starting_data = XX.media_data.copy()
        target_func = _target_function_media
    elif vars_type == 'non-media':
        starting_data = _inverse_scale(XX.non_media_data, *XX.NonMediaMinMaxCenter())
        target_func = _target_function_non_media

    jax.config.update("jax_enable_x64", True)

    partial_target_function = functools.partial(
        target_func, 
        vars_to_adjust=vars_to_adjust, 
        period_to_adjust=period_to_adjust, 
        target_value=target_value, 
        X=XX, 
        starting_data=starting_data, 
        modeller=modeller)

    return scipy.optimize.minimize(fun=partial_target_function,
                                       x0=starting_values, 
                                       method="SLSQP",
                                       jac="3-point", 
                                       bounds=bounds,
                                       options={
                                          "maxiter": 200,
                                          "disp": True,
                                          "ftol": 1e-06,
                                          "eps": 1.4901161193847656e-08,
                                          }
                                      )

############# target reach end ############# 
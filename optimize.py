import scipy.optimize
import functools
import jax

from .definitions import *
from .modeller import Modeller
from .sales_model import SalesModel 

from typing import Dict, List, Tuple

############# media optimization utils ############# 

def _get_time_index(period: int | list | tuple, df: pd.DataFrame) -> slice:
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

def _get_var_index(vars_: list | str, modeller: Modeller) -> list:
    """
    проверяет на ошибки в списке
    принимает как строку (одна переменная), так и список
    """
    # прин
    if isinstance(vars_, str): 
        vars_ = [vars_]
    var_index = [i for i, (_, v) in enumerate(modeller.X.MediaVarnames()) if v in vars_]
    if len(var_index) > 0: 
        if len(var_index) != len(vars_):
            raise ValueError("check variables to optimize. They must all be either from 'media' or 'non-media'")
        return 'media', tuple(var_index)
    
    var_index = [i for i, (_, v) in enumerate(modeller.X.NonMediaVarnames()) if v in vars_]
    if len(var_index) > 0: 
        if len(var_index) != len(vars_):
            raise ValueError("check variables to optimize. They must all be either from 'media' or 'non-media'")
        return 'non-media', tuple(var_index)


############# media optimization begin ############# 
def _check_media_optimize_spec(opt_index, X, keep_pattern): 
    TOLERANCE = 0.05
    if keep_pattern:
        channel_shares = X.media_data[opt_index].sum(axis=0) / X.media_data[opt_index].sum(axis=None)
        assert (channel_shares > TOLERANCE).all(),\
            "Not enough spends for the period to keep pattern. Check shares {}".format(channel_shares)


def _get_starting_allocation(opt_index: tuple, X) -> jnp.array: 
    sum_ = X.media_data[opt_index].sum(axis=0)
    return sum_ / sum_.sum()

def _constraints(alloc): 
    return jnp.sum(alloc) - 1

@functools.partial(
    jax.jit,
    static_argnames=("prices", "budget"))
def _budget_constraints(alloc, prices, budget): 
    return jnp.sum(alloc * jnp.array(prices)) - budget

def _get_optimization_bounds(historical_values, prices) -> scipy.optimize.Bounds:
    media_prices = jnp.array(prices)
    budget = (historical_values * media_prices).sum()
    return scipy.optimize.Bounds(jnp.zeros_like(historical_values), budget / media_prices)

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


def CalculateGains(df: pd.DataFrame,
                   media_to_optimize: list[str],
                   period_to_optimize: int | list | None,
                   media_allocation: list,
                   modeller: Modeller, 
                   keep_spend_pattern: bool = True) -> float:
    opt_index = (
        _get_time_index(period_to_optimize, df), 
        _get_var_index(media_to_optimize, modeller)[1]
    )
    
    X = modeller.PrepareNewCovs(df)
    starting_allocation = _get_starting_allocation(opt_index, X)
    if keep_spend_pattern == False:
        X.media_data = X.media_data.at[opt_index].set(_redistribute_evenly(X.media_data[opt_index], starting_allocation))
    y_before = modeller.Predict(X, return_decomposition=False)['y'][opt_index[0]].sum()
    
    X = modeller.PrepareNewCovs(df)
    if keep_spend_pattern:
        X.media_data = X.media_data.at[opt_index].set(_redistribute_keep_pattern(X.media_data[opt_index], jnp.array(media_allocation)))
    else: 
        X.media_data = X.media_data.at[opt_index].set(_redistribute_evenly(X.media_data[opt_index], jnp.array(media_allocation)))
    
    y_after = modeller.Predict(X, return_decomposition=False)['y'][opt_index[0]].sum()
    print("Target metric...:")
    print("...before: {}".format(y_before))
    print("...after: {}".format(y_after))
    print("Gain: {}".format(y_after / y_before - 1))
    return y_after / y_before - 1
   
def OptimizeMediaAllocation(df: pd.DataFrame, 
             media_to_optimize: list[str],
             period_to_optimize: int | list | None,
             modeller: Modeller, 
             media_prices: list = None,
             keep_spend_pattern: bool = True):
    """
    df: dataframe to work with
    media_to_optimize: media-vars to manipulate with: list of names
    period_to_optimize: period to manipulate with and watch target. int - last x datapoints / list - from to / None - all 
    modeller: fitted model
    keep_spend_pattern: True - keep / False - redistribute evenly across period
    """
    opt_index = (
        _get_time_index(period_to_optimize, df), 
        _get_var_index(media_to_optimize, modeller)[1]
    )

    print("Variable index {}".format(opt_index[1]))
    
    X = modeller.PrepareNewCovs(df)
    _check_media_optimize_spec(opt_index, X, keep_spend_pattern)

    if media_prices is None: 
        media_prices = jnp.ones_like(media_to_optimize)
    else: 
        assert len(media_to_optimize) == len(media_prices), "Expected: len(media_to_optimize) == len(media_prices)"
        #media_prices = jnp.array(media_prices)

    # getting starting values as historical means, not from new data
    starting_allocation = _get_starting_allocation(opt_index, modeller.X)
    media_budget = (starting_allocation * jnp.array(media_prices)).sum()
    bounds = _get_optimization_bounds(starting_allocation, media_prices)
    print("Starting allocation: {}, {}".format(starting_allocation, bounds))

    jax.config.update("jax_enable_x64", True)

    partial_objective_function = functools.partial(
        _objective_function, 
        X=X, opt_index=opt_index, modeller=modeller, keep_spend_pattern=keep_spend_pattern)
    
    constraints_function = functools.partial(
        _budget_constraints, 
        prices=media_prices, budget=float(media_budget)
    )

    opt_result = scipy.optimize.minimize(fun=partial_objective_function,
                                       x0=starting_allocation, 
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
                                          "fun": constraints_function
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
                varnames_to_adjust: list[int],
                modeller: Modeller,
                adjust_bounds: float=0.5):
    """
    df: dataframe to work with
    relative_target: 1.1 means 10% increase vs. current etc.
    period_to_adjust: period to manipulate with and watch target. int - last x datapoints / list - from to / None - all 
    varnames_to_adjust: vars to manipulate with: list of names
    modeller: fitted model
    """
    
    period_to_adjust = _get_time_index(period_to_adjust, df)
    vars_type, vars_to_adjust = _get_var_index(varnames_to_adjust, modeller)
    print("Variable type: {}, variable index{}".format(vars_type, vars_to_adjust))
    
    XX = modeller.PrepareNewCovs(df)
    current_value = modeller.model.PredictY(XX)['y'].mean(axis=0)[period_to_adjust].sum()
    target_value = float(relative_target * current_value)
    starting_values = jnp.ones(1)
    bounds = scipy.optimize.Bounds(starting_values * (1.0 - adjust_bounds), starting_values * (1.0 + adjust_bounds))
    print("Current value: {}, Target value: {} (non-inverse-scaled)".format(current_value, target_value))
    print("Starting values: {}, {}".format(starting_values, bounds))

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


############# variables elasticity utils ############# 

def _get_multipliers(mults: Dict, modeller) -> Tuple[jnp.array]:
    """
    перерабатывает словарь "имя переменной > множитель" в массивы множителей полной размерности 
    """
    
    media_var_index = modeller.X.MediaVarIndex()
    nonmedia_var_index = modeller.X.NonMediaVarIndex()

    media_mults = [1.0] * len(media_var_index)
    nonmedia_mults = [1.0] * len(nonmedia_var_index)

    for v, m in mults.items():
        if v in media_var_index:
            media_mults[media_var_index[v]] = m
        elif v in nonmedia_var_index:
            nonmedia_mults[nonmedia_var_index[v]] = m
        else:
            raise ValueError("Unknown variable {}".format(v))
    
    return jnp.array(media_mults), jnp.array(nonmedia_mults)


def MultCovsCalculateTraget(
                   df: pd.DataFrame,
                   modeller: Modeller,
                   multipliers: Dict,
                   period_to_apply: int | list | None,
                   keep_time_pattern: bool = True) -> float:
    """
    на вход подается словарь множителей для каждой независимой переменной, и период к которому они применяютя
    делает прогноз зависимой переменной, считает ее суммарное значение за тот же период 
    
    df: данные с независимыми переменными, необязательно те, на которых строилас модель 
    modeller: обученная модель 
    multipliers: словарь множителей
    period_to_apply: период для которого применяются множители 
    keep_time_pattern: True - не меняет распределение Х по времени / False - меняет
    """
    mult_media, mult_nonmedia = _get_multipliers(multipliers, modeller) 
    time_index = _get_time_index(period_to_apply, df)

    X = modeller.PrepareNewCovs(df)
    if keep_time_pattern:
        X.media_data = X.media_data.at[time_index].set(X.media_data[time_index] * mult_media)
        X.non_media_data = X.non_media_data.at[time_index].set(X.non_media_data[time_index] * mult_nonmedia)
    else: 
        raise NotImplementedError()

    return modeller.Predict(X, return_decomposition=False)['y'][time_index].sum()

def ElasticityByVariable(
        df: pd.DataFrame,
        modeller: Modeller,
        varible: str,
        period_to_apply: int | List | None,
        elasricyty_range: List[float] = [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5],
        keep_time_pattern: bool = True) -> float:
    
    target_values = []
    for val in elasricyty_range:
        target_values.append(
            MultCovsCalculateTraget(
                df=df, modeller=modeller,
                multipliers={varible: val},
                period_to_apply=period_to_apply,
                keep_time_pattern=keep_time_pattern
            )
        )
    return pd.concat([pd.Series(elasricyty_range), pd.Series(target_values)], axis=1)




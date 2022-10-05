import numpy as np
import time
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt


def sizeof_fmt(num, suffix='Gi'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


class TikTok:
    """Class to monitor time"""

    def __init__(self):
        self.start = time.time()
        self.last = self.start
        self.laps = {}

    def step(self, msg="", display=True):
        if display:
            print(f'Step: {msg} \n'
                  f'Last: {time.time() - self.last: .1f} s \n'  # on peut mettre : dans une accolade pour dire comment on formatte
                  f'Begin: {time.time() - self.start: .1f} s')  # une décimale en flottant
        self.last = time.time()

    def reset(self):
        self.start = time.time()

    def interval(self, key, display=True):
        if display:
            if key in self.laps:
                print(f"Time to calculate {key}: {time.time() - self.laps[key]: .1f} s")
        self.laps[key] = time.time()

    def __call__(self, *args, **kwargs):  # quand on appelle TIKTOK(), cela applique la fonction step
        self.step(*args, **kwargs)


TIKTOK = TikTok()


def supply_function_np_piecewise(x: np.ndarray, x_cutoff: np.ndarray, y_cutoff: np.ndarray, voll):
    """
    Returns price corresponding to residual demand. Piecewise affine function.
    Parameters
    ----------
    x: residual demand
    x_cutoff: cutoff points corresponding to cumulative capacity
    y_cutoff: cutoff points corresponding to marginal price
    voll: value of loss load

    Returns
    -------

    """
    x = x.astype(np.float32)  # we avoid pathological case where x is an int (the result would then be an int as well)
    return np.piecewise(
        x,
        [(x >= x_cutoff[i]) & (x < x_cutoff[i + 1]) for i in range(len(x_cutoff) - 1)] + [x >= x_cutoff[-1]],
        [
            (lambda y, k=i: ((y_cutoff[k + 1] - y_cutoff[k]) * (y - x_cutoff[k]) / (x_cutoff[k + 1] - x_cutoff[k]) +
                             y_cutoff[k]))
            for i in range(len(x_cutoff) - 1)
        ] + [lambda y: voll],
    )


def supply_function_np_piecewise_constant(x: np.ndarray, x_cutoff: np.ndarray, y_cutoff: np.ndarray, voll):
    """
    Returns price corresponding to residual demand. Piecewise constanat function.
    Parameters
    ----------
    x: residual demand
    x_cutoff: cutoff points corresponding to cumulative capacity
    y_cutoff: cutoff points corresponding to marginal price
    voll: value of loss load

    Returns
    -------

    """
    x = x.astype(np.float32)  # we avoid pathological case where x is an int (the result would then be an int as well)
    return np.piecewise(
        x,
        [(x >= x_cutoff[i]) & (x < x_cutoff[i + 1]) for i in range(len(x_cutoff) - 1)] + [x >= x_cutoff[-1]],
        [
            (lambda y, k=i: y_cutoff[k + 1])
            for i in range(len(x_cutoff) - 1)
        ] + [lambda y: voll],
    )


def price_function(Q: np.ndarray, D_average: np.ndarray, weather_params, Q_offshore,
                   x_cutoff, y_cutoff, add_params):
    """
    Returns price corresponding to a given demand and renewable production
    Parameters
    ----------
    Q: installed capacity
    D_average: average demand at time t
    D_centered: possible values for centered demand
    gamma_sun: possible values for sun capacity factor
    gamma_wind_onshore: possible values for wind capacity factor

    Returns
    -------
    La fonction marche bien a priori !!
    """
    if type(D_average) != np.ndarray:
        D_average = np.array([D_average])
    assert D_average.ndim == 1  # attention il faut changer le reshape si D_average prend plusieurs dimensions dans le
    # futur
    assert Q.ndim > 1  # first dimension corresponds to Q_sun and Q_wind, other dimensions correspond to the number of
    # considered paths
    # Q: (2,d,...,d) where we repeat d (t+1) times (corresponding to the possible paths)
    # Demand : (d,)
    t = Q.ndim - 1  # we remove dimension associated to the choice between Q_sun and Q_wind
    market_price, voll, Q_pv_init, Q_onshore_init, Q_river = (
        add_params['market_price'], add_params['voll'], add_params['Q_pv_init'], add_params['Q_onshore_init'],
        add_params["Q_river"])
    demand_centered, gamma_sun, gamma_wind_onshore, gamma_wind_offshore, gamma_river, hydro_prod = (
        weather_params["demand"], weather_params["sun"], weather_params["onshore"], weather_params["offshore"],
        weather_params["river"], weather_params["hydro"])
    assert demand_centered.shape == gamma_sun.shape == gamma_wind_onshore.shape  # shape is (n,) where n is the number of
    # discretized steps
    return np.minimum(supply_function_np_piecewise_constant(np.maximum(
        D_average.reshape((1,) * (t + 1) + (-1,)) + demand_centered.reshape((-1,) + (1,) * (t + 1))
        - Q[0].reshape((1,) + Q[0].shape + (1,)) * gamma_sun.reshape((-1,) + (1,) * (t + 1))
        - Q[1].reshape((1,) + Q[1].shape + (1,)) * gamma_wind_onshore.reshape((-1,) + (1,) * (t + 1))
        - Q_pv_init * gamma_sun.reshape((-1,) + (1,) * (t + 1))
        - Q_onshore_init * gamma_wind_onshore.reshape((-1,) + (1,) * (t + 1))
        - Q_offshore * gamma_wind_offshore.reshape((-1,) + (1,) * (t + 1))
        - Q_river * gamma_river.reshape((-1,) + (1,) * (t + 1))
        - hydro_prod.reshape((-1,) + (1,) * (t + 1)), 0
    ), np.array(x_cutoff), np.array(y_cutoff), voll
    ), market_price)  # shape (n,) + (d,)*(t+1) where the t first d corresponds to allowed paths, and the last d
    # correspond to D_average


def coefficient_f(Q: np.ndarray, D_average, premium, weather_params,
                  Q_offshore, tec, x_cutoff, y_cutoff, add_params):
    """Returns coefficient f corresponding to expected profit over the given time interval, depending on the
    considered technology"""
    # assert isinstance(premium, int) or isinstance(premium, float)  # peut etre a changer si on considère des arrays
    Delta_t = add_params["Delta_t"]
    gamma_sun, gamma_wind_onshore, proba = (
        weather_params["sun"], weather_params["onshore"], weather_params["proba"])
    t = Q.ndim - 1  # this corresponds to state Q at time t-1, since Q^{sun} will be of shape t
    premium = np.array([premium])  # shape (1,)
    price = price_function(Q, D_average, weather_params, Q_offshore, x_cutoff, y_cutoff,
                           add_params)  # shape (n,)+(d,)*(t+1)
    price = price + premium  # we add a market premium which is the same whatever the price and weather conditions
    if tec == 'sun':
        profit_sun = price * gamma_sun.reshape((-1,) + (1,) * (t + 1))
        return ((proba.reshape((-1,) + (1,) * (t + 1)) * profit_sun).sum(0)) * Delta_t  # shape (d,)*(t+1)
    else:  # tec == 'wind'
        profit_wind = price * gamma_wind_onshore.reshape((-1,) + (1,) * (t + 1))
        return (proba.reshape((-1,) + (1,) * (t + 1)) * profit_wind).sum(0) * Delta_t  # shape (d,)*(t+1)
        # should be ok


def concatenate_coef_f(Q, demand_states_time_t, premium, weather_params, Q_offshore, tec, x_cutoff, y_cutoff,
                       add_params):
    """Compiling coefficient_f for different values of demand and the corresponding values of the cutoff
    points, as well as the offshore capacity values."""
    coef_f_values = coefficient_f(Q, np.array([demand_states_time_t[0]]), premium, weather_params,
                                  Q_offshore[0], tec, x_cutoff[0], y_cutoff[0],
                                  add_params)  # first initialize for the first value of demand
    for i in range(1, demand_states_time_t.shape[0],
                   1):  # iterate for all possible values of demand and corresponding cutoffs
        coef_f_values_demand = coefficient_f(Q, np.array([demand_states_time_t[i]]), premium, weather_params,
                                             Q_offshore[i], tec, x_cutoff[i], y_cutoff[i], add_params)
        coef_f_values = np.concatenate((coef_f_values, coef_f_values_demand), axis=-1)  # we only concatenate along
        # the final axis corresponding to uncertain demand + mix
        # TODO: je peux simplifier ce passage en concatenant juste à la fin
    return coef_f_values


def price_function_cvar(Q: np.ndarray, D_cvar: np.ndarray, weather_tot_params, Q_offshore,
                        x_cutoff, y_cutoff, add_params):
    """Adaptation of price_function to CVAR. In this function, we consider that D_cvar corresponds to a single scenario.
     The matching with the different paths for Q will be done in another function. This explains how the rematching is
      done in this specific function."""
    demand_centered, gamma_sun, gamma_wind_onshore, gamma_wind_offshore, gamma_river, hydro_prod = (
        weather_tot_params["demand"], weather_tot_params["sun"], weather_tot_params["onshore"],
        weather_tot_params["offshore"], weather_tot_params["river"], weather_tot_params["hydro"])
    assert demand_centered.shape == gamma_sun.shape == gamma_wind_onshore.shape  # shape is (n,) where n is the number of
    # discretized steps
    assert D_cvar.ndim == 1  # on vérifie qu'on considère une seule valeur de D_cvar
    Q_0, Q_1 = np.array(Q[0]), np.array(Q[1])
    # Here, in practice, Q would correspond to Q_{t+1}, and we would only correspond the first t dimensions of Q_{t+1}
    if type(D_cvar) != np.ndarray:
        D_cvar = np.array([D_cvar])
    market_price, voll, Q_pv_init, Q_onshore_init, Q_river = add_params['market_price'], add_params['voll'], \
                                                             add_params['Q_pv_init'], add_params[
                                                                 'Q_onshore_init'], add_params["Q_river"]
    return np.minimum(supply_function_np_piecewise_constant(np.maximum(
        D_cvar.reshape((1,) * Q.ndim) + demand_centered.reshape((-1,) + (1,) * (Q.ndim - 1))
        - Q_0.reshape((1,) + Q_0.shape) * gamma_sun.reshape((-1,) + (1,) * (Q.ndim - 1))
        - Q_1.reshape((1,) + Q_1.shape) * gamma_wind_onshore.reshape((-1,) + (1,) * (Q.ndim - 1))
        - Q_pv_init * gamma_sun.reshape((-1,) + (1,) * (Q.ndim - 1))
        - Q_onshore_init * gamma_wind_onshore.reshape((-1,) + (1,) * (Q.ndim - 1))
        - Q_offshore * gamma_wind_offshore.reshape((-1,) + (1,) * (Q.ndim - 1))
        - Q_river * gamma_river.reshape((-1,) + (1,) * (Q.ndim - 1))
        - hydro_prod.reshape((-1,) + (1,) * (Q.ndim - 1)), 0
    ), np.array(x_cutoff), np.array(y_cutoff), voll
    ),
        market_price)  # shape (n,) + (d,)*t where the t d corresponds to paths for variable Q


def coefficient_f_cvar(Q: np.ndarray, D_cvar, premium,
                       weather_tot_params, Q_offshore, tec, x_cutoff, y_cutoff, add_params):
    """Returns an array of shape (n,) + (d,)*(t+1) which represents CVAR profit, if we consider that Q is taken at time
    t. We take into account the path which determines the demand/mix corresponding to CVAR. We note that we have to use
     a for loop here, as the supply function also depends on the CVAR. We cannot use directly a broadcast operation. So
     what we do is that we fix the last coordinate for Q, and consider the corresponding CVAR demand.
     REMARK; we consider that the cutoff points x_cutoff have been selected to match the cvar scenario considered."""
    # REMARK: here, we consider that state Q is taken at time t
    gamma_sun, gamma_wind_onshore, proba = (
        weather_tot_params["sun"], weather_tot_params["onshore"],
        weather_tot_params["proba"])
    assert len(x_cutoff) == len(y_cutoff) == D_cvar.shape[0] == Q.shape[-1]  # we check that the number of
    # scenarios for demand/mix are the same for all variables which depend on them
    premium = np.array([premium])  # shape (1,)
    price = price_function_cvar(Q[..., 0], np.array([D_cvar[0]]), weather_tot_params, Q_offshore[0],
                                x_cutoff[0], y_cutoff[0], add_params)  # shape (n,) + (d,)*(t-1)
    price = price.reshape(price.shape + (1,))  # we add a final axis corresponding to the last coordinate of Q
    # TODO: remarque, une autre solution serait de garder un axe dans price_function_cvar pour la demande
    for i in range(1, D_cvar.shape[0], 1):
        price_demand = price_function_cvar(Q[..., i], np.array([D_cvar[i]]), weather_tot_params, Q_offshore[i],
                                           x_cutoff[i], y_cutoff[i], add_params)  # shape (n,) + (d,)*t
        price_demand = price_demand.reshape(price_demand.shape + (1,))
        price = np.concatenate((price, price_demand), axis=-1)  # we concatenate along the final axis, corresponding
        # to the value of demand at time t-2 (if we consider Q taken at time t-1)
        # TODO: same here, we can maybe speed up the code by concatenating only at the end
    price = price + premium  # shape (n,) + (d,)*(t+1)
    if tec == 'sun':
        price_tec = price * gamma_sun.reshape((-1,) + (1,) * (price.ndim - 1))
    else:  # wind
        price_tec = price * gamma_wind_onshore.reshape((-1,) + (1,) * (price.ndim - 1))

    profit = price_tec * proba.reshape((-1,) + (1,) * (price.ndim - 1))  # shape (n,) + (d,)*(t+1)
    return profit


def calculate_cvar_par(alpha, start, end, gas_scenarios, proba_worst_demand, Q: np.ndarray, D_cvar, premium,
                       weather_tot_params, Q_offshore, tec, x_cutoff, y_cutoff, add_params):
    """
    Returns the CVAR corresponding to coefficient f for a given level alpha, using parallel computing.
     Uncertainty comes from different gas price scenarios, and from different weather data years.
    REMARK: we still consider that the cutoff points have been chosen correctly to match the demand cvar scenarios.
     This is a preprocessing step.
    Parameters
    ----------
    start: int
        Start year to consider when calculating cvar
    end: int
        End year to consider when calculating cvar
    gas_scenarios

    Returns
    -------

    """
    # We consider that, as for coef_f, Q is at time t-1, and therefore has t dimensions
    Delta_t = add_params["Delta_t"]
    years = weather_tot_params["years"]
    premium = np.array([premium])  # shape (1,)

    def in_loop(i):
        coef_f_values_list = []
        scenario = gas_scenarios[i]
        y_cutoff_scenario = []
        for k in range(len(y_cutoff)):  # we modify the gas price for each cutoff corresponding to demand/mix scenario
            y_cutoff_gas = y_cutoff[k][
                           2:]  # TODO: pour l'instant, j'augmente également le prix du charbon, sans doute une approximation pas très grave
            y_cutoff_gas = list(y_cutoff_gas + scenario)
            y_cutoff_k_scenario = y_cutoff[k][0:2] + y_cutoff_gas
            y_cutoff_scenario.append(y_cutoff_k_scenario)
        profit = coefficient_f_cvar(Q, D_cvar, premium, weather_tot_params, Q_offshore, tec, x_cutoff,
                                    y_cutoff_scenario, add_params)  # shape (n,) + (d,)*(t+1)
        for y in np.arange(start, end + 1, 1):
            mask = (years == y)
            coef_f = profit[mask, ...].sum(0) * Delta_t  # shape (d,)*(t+1)
            coef_f = coef_f.reshape(coef_f.shape + (1,))  # we add a final axis corresponding to the scenario
            coef_f_values_list.append(coef_f)
        coef_f_values = np.concatenate(coef_f_values_list, axis=-1)
        return coef_f_values

    pool = Parallel(n_jobs=8, prefer="threads")
    coef_f_values = pool(delayed(in_loop)(i) for i in range(gas_scenarios.shape[0]))
    del pool  # TODO: verifier que c'est fait correctement, pas évident
    coef_f_values = np.concatenate(coef_f_values, axis=-1)
    # coef_f_values has shape (d,)*(t+1) + (M,) where M is the total number of scenarios
    coef_f_values = np.sort(coef_f_values,
                            axis=-1)  # we sort by increasing value according to the last axis, corresponding to scenarios
    nb_scenarios_cvar = int(np.round(coef_f_values.shape[-1] * alpha))
    coef_f_values_cvar = np.stack(
        [coef_f_values[..., i, 0:int(nb_scenarios_cvar * 1 / p)].mean(axis=-1) for i, p in
         enumerate(proba_worst_demand)],
        axis=-1)  # peut etre problème avec un max, si jamais on obtenait un nombre supérieur à la dernière dimension
    return coef_f_values_cvar  # shape (d,)*(t+1)


def coefficient_g(T, Tprime, Q: np.ndarray, D_average, premium, weather_params, Q_offshore, tec, x_cutoff, y_cutoff,
                  add_params):
    """Returns coefficient g representing the end of the profit after time T, which depends
    on the value of demand at time T-1. Q represents invested capacity at time T (and not time T-1 !!)"""
    assert Q.ndim - 1 == T  # corresponds to investment decision at time T-1 (after being devaluated once), plus one
    # dimension for sun and wind
    discount_rate, nu_deval = add_params['discount_rate'], add_params['nu_deval']
    if Tprime > T:
        coef_g = - coefficient_f(Q, D_average, premium, weather_params, Q_offshore, tec, x_cutoff, y_cutoff, add_params)
        for t in range(T + 1, Tprime, 1):
            coef_g += -np.exp(-discount_rate * (t - T)) * (1 - nu_deval) ** (t - T) * coefficient_f(
                Q * (1 - nu_deval) ** (t - T), D_average, premium, weather_params, Q_offshore, tec,
                x_cutoff, y_cutoff, add_params)  # shape (d,)*(T+1)
    else:
        coef_g = np.zeros((D_average.shape[0],) * (T + 1))  # adding option that we do not account for further profit
    return coef_g


def coefficient_g_no_deval(T, Tprime, Q: np.ndarray, D_average, premium, weather_params, Q_offshore, tec, x_cutoff, y_cutoff,
                  add_params):
    """Returns coefficient g representing the end of the profit after time T, which depends
    on the value of demand at time T-1. Q represents invested capacity at time T (and not time T-1 !!)"""
    assert Q.ndim - 1 == T  # corresponds to investment decision at time T-1 (after being devaluated once), plus one
    # dimension for sun and wind
    discount_rate, nu_deval = add_params['discount_rate'], add_params['nu_deval']
    if Tprime > T:
        coef_g = - coefficient_f(Q, D_average, premium, weather_params, Q_offshore, tec, x_cutoff, y_cutoff, add_params)
        for t in range(T + 1, Tprime, 1):
            coef_g += -np.exp(-discount_rate * (t - T)) * coefficient_f(Q, D_average, premium, weather_params,
                                                                         Q_offshore, tec, x_cutoff, y_cutoff,
                                                                         add_params)  # shape (d,)*(T+1)
    else:
        coef_g = np.zeros((D_average.shape[0],) * (T + 1))  # adding option that we do not account for further profit
    return coef_g


def concatenate_coef_g(T, Tprime, Q, demand_states_time_T, premium, weather_params, Q_offshore, tec, x_cutoff, y_cutoff,
                       add_params):
    """Compiling coefficient_g for different values of demand, to allow for the simultaneous evolution of the cutoff
    points as well as the offshore values."""
    coef_g_values = coefficient_g(T, Tprime, Q, np.array([demand_states_time_T[0]]), premium, weather_params,
                                  Q_offshore[0], tec, x_cutoff[0], y_cutoff[0],
                                  add_params)  # first initialize for the first value of demand
    for i in range(1, demand_states_time_T.shape[0], 1):  # iterate for all possible values of demand and cutoffs
        coef_g_values_demand = coefficient_g(T, Tprime, Q, np.array([demand_states_time_T[i]]), premium, weather_params,
                                             Q_offshore[i], tec, x_cutoff[i], y_cutoff[i], add_params)
        coef_g_values = np.concatenate((coef_g_values, coef_g_values_demand), axis=-1)
    return coef_g_values  # shape (d,)*(T+1)


def concatenate_coef_g_no_deval(T, Tprime, Q, demand_states_time_T, premium, weather_params, Q_offshore, tec, x_cutoff, y_cutoff,
                       add_params):
    """Compiling coefficient_g for different values of demand, to allow for the simultaneous evolution of the cutoff
    points as well as the offshore values."""
    coef_g_values = coefficient_g_no_deval(T, Tprime, Q, np.array([demand_states_time_T[0]]), premium, weather_params,
                                  Q_offshore[0], tec, x_cutoff[0], y_cutoff[0],
                                  add_params)  # first initialize for the first value of demand
    for i in range(1, demand_states_time_T.shape[0], 1):  # iterate for all possible values of demand and cutoffs
        coef_g_values_demand = coefficient_g_no_deval(T, Tprime, Q, np.array([demand_states_time_T[i]]), premium, weather_params,
                                             Q_offshore[i], tec, x_cutoff[i], y_cutoff[i], add_params)
        coef_g_values = np.concatenate((coef_g_values, coef_g_values_demand), axis=-1)
    return coef_g_values  # shape (d,)*(T+1)


def matrix_for_cvar(trans_matrix):
    """Returns transition matrix associated to cvar. This matrix indicates the transition to the worst case scenario
    in terms of demand, for each possible value of demand at time t. It works because we coded transition matrix so that
    state 0 corresponds to the lowest value for demand. It works also because we only consider uncertainty on the level
    of demand. Not useful when we add the other sources of uncertainty."""
    n, m = trans_matrix.shape
    trans_matrix_cvar = np.zeros((n, m))
    ind = (trans_matrix != 0).argmax(axis=1)
    trans_matrix_cvar[range(trans_matrix.shape[0]), ind] = 1
    assert np.allclose(trans_matrix_cvar.sum(1),
                       np.ones(n))  # we check that we have something summing up to one in each line
    return trans_matrix_cvar


def probability_worst_demand(trans_matrix):
    """Returns the probability corresponding to the worst value for each initial value of demand. Ie, we take
    the first non zero value in trans_matrix per line."""
    return trans_matrix[range(trans_matrix.shape[0]), (trans_matrix != 0).argmax(axis=1)]  # shape (d,)


def index_cvar(trans_matrix):
    return (trans_matrix != 0).argmax(axis=1)


def recursive_optimal_control(t, q_next, trans_matrix, demand_states, list_beta, alpha, start, end, gas_scenarios,
                              Q_t: np.ndarray, premium, weather_params, weather_tot_params,
                              Q_offshore_t, tec, x_cutoff_t, y_cutoff_t, add_params):
    """ NEW VERSION WITH UNCERTAINTY ON GAS. Returns optimal control at time t, of shape (d,)*(t+1)
    Here, we can obtain negative values. The thresholding to obtain positive values is only applied at the end
    of the recursive process """
    discount_rate, nu_deval, c_tilde = add_params['discount_rate'], add_params['nu_deval'], add_params[f"c_tilde_{tec}"]
    assert tec in ['sun', 'wind'], f"tec should be equal to sun or wind, but instead is equal to {tec}"
    assert t < trans_matrix.shape[0] - 1, "Time step t does not have a corresponding transition matrix"  # This
    # function works for t < T-1 (for T-1, we have a specific initialization function)
    assert type(Q_t) == type(q_next) == np.ndarray, "Q and q_next should be of array type"
    trans_matrix_t = trans_matrix[t, :, :]  # shape (d,d)
    assert q_next.ndim == t + 2 + 1, "Parameter q_next should have t+2 dimensions corresponding to all possible paths until" \
                                 "time t+1 and a first dimension corresponding to parameter beta "
    assert Q_t.ndim == t + 2, "Parameter Q should have t+2 dimensions corresponding to all possible paths until time t," \
                              "for both wind and solar "
    assert q_next.shape[-1] == trans_matrix_t.shape[-1], "Parameter q_next should have the same last dimension as the " \
                                                         "second dimension of trans_matrix_t "
    assert q_next.shape[-2] == trans_matrix_t.shape[0], "Parameter q_next should have the same second last dimension " \
                                                        "as the first dimension of trans_matrix_t "
    assert Q_t.shape[-1] == trans_matrix_t.shape[0], "Last dimension of parameter Q should be the same as the first " \
                                                     "dimension of trans_matrix_t "
    expectation_qnext = (q_next * trans_matrix_t).sum(-1)  # shape (d,)*(t+1)
    assert expectation_qnext.ndim == t + 1 + 1, "Intermediate calculations to check, wrong dimension obtained, maybe problem with list_beta"
    # Natural broadcasting should work as expected, as Numpy starts with the rightmost dimensions and works its way left
    D_average_t = demand_states[t, :]  # possible value of demand during interval [t:t+1]
    assert D_average_t.shape[0] == q_next.shape[
        -1], "Dimension of possible demands at time t,t+1 should be the same as " \
             "last dimension of future strategy q_{next}"
    coef_f_values = concatenate_coef_f(Q_t, D_average_t, premium, weather_params, Q_offshore_t, tec,
                                       x_cutoff_t, y_cutoff_t, add_params)  # shape (d,)*(t+2)
    expectation_coefficient_f = (coef_f_values * trans_matrix_t).sum(-1)  # shape (d,)*(t+1)
    assert coef_f_values.ndim == t + 2
    trans_matrix_t_cvar = matrix_for_cvar(trans_matrix_t)
    D_cvar_t = (D_average_t * trans_matrix_t_cvar).sum(-1)  # shape (d,)
    proba_worst_demand = probability_worst_demand(trans_matrix_t)  # shape (d,)
    ind_cvar = index_cvar(trans_matrix_t)
    x_cutoff_t_cvar = [x_cutoff_t[i] for i in ind_cvar]
    y_cutoff_t_cvar = [y_cutoff_t[i] for i in ind_cvar]
    Q_offshore_t_cvar = [Q_offshore_t[i] for i in ind_cvar]
    # IMPORTANT REMARK: here, we consider that it is only the worst demand scenario at a given point t which determines
    # the CVAR. This is an approximation !!
    cvar_coefficient_f = calculate_cvar_par(alpha, start, end, gas_scenarios, proba_worst_demand, Q_t, D_cvar_t,
                                            premium, weather_tot_params, Q_offshore_t_cvar, tec, x_cutoff_t_cvar,
                                            y_cutoff_t_cvar, add_params)  # shape (d,)*(t+1)

    assert expectation_coefficient_f.shape == cvar_coefficient_f.shape

    expectation_coefficient_f_tilde_beta = []
    for beta in list_beta:  # we calculate expectation_coefficient_f_tilde for all beta values
        expectation_coefficient_f_tilde = beta * expectation_coefficient_f + (1 - beta) * cvar_coefficient_f
        expectation_coefficient_f_tilde = expectation_coefficient_f_tilde.reshape((1,) + expectation_coefficient_f_tilde.shape)  # we add a first dimension corresponding to parameter beta
        expectation_coefficient_f_tilde_beta.append(expectation_coefficient_f_tilde)

    expectation_coefficient_f_tilde = np.concatenate(expectation_coefficient_f_tilde_beta, axis=0)  # concatenate along first axis

    investment_costs = add_params[f'investment_costs_{tec}']
    Delta_investment_costs = investment_costs[t] - (1 - nu_deval) * np.exp(-discount_rate) * investment_costs[
        t + 1]  # should be positive
    q_t = 1 / (2 * c_tilde) * (expectation_coefficient_f_tilde - Delta_investment_costs) + (1 - nu_deval) * np.exp(
        -discount_rate) * expectation_qnext  # shape (L,) + (d,)*(t+1)
    assert q_t.ndim == t + 1 + 1  # we added a dimension for beta
    return q_t


def optimal_control_final(T, Tprime, trans_matrix, demand_states, list_beta, alpha, start, end, gas_scenarios, Q_T_1,
                          premium, weather_params, weather_tot_params, Q_offshore_T_1, tec, x_cutoff_T_1,
                          y_cutoff_T_1, add_params):
    """NEW VERSION. Finds the value of optimal control final. Q corresponds to investment decisions by other agents at time
    T-1. Shape (d,)*(T)"""
    assert T == trans_matrix.shape[0]
    discount_rate, nu_deval, c_tilde = add_params['discount_rate'], add_params['nu_deval'], add_params[f'c_tilde_{tec}']
    trans_matrix_T_1 = trans_matrix[T - 1]  # shape (d,d)
    D_average_T_1 = demand_states[T - 1, :]  # shape (d,)
    coef_f_values = concatenate_coef_f(Q_T_1, D_average_T_1, premium, weather_params, Q_offshore_T_1, tec,
                                       x_cutoff_T_1, y_cutoff_T_1, add_params)  # shape (d,)*(T+1)

    assert coef_f_values.ndim == T + 1
    expectation_coefficient_f = (coef_f_values * trans_matrix_T_1).sum(-1)  # shape (d,)*(T)
    # On prend en compte la CVAR pour la partie des profits entre T-1 et T, mais pas après
    trans_matrix_T_1_cvar = matrix_for_cvar(trans_matrix_T_1)
    D_cvar_T_1 = (D_average_T_1 * trans_matrix_T_1_cvar).sum(-1)  # shape (d,)*(t+1)
    proba_worst_demand = probability_worst_demand(trans_matrix_T_1)  # shape (d,)
    ind_cvar = index_cvar(trans_matrix_T_1)
    x_cutoff_T_1_cvar = [x_cutoff_T_1[i] for i in ind_cvar]
    y_cutoff_T_1_cvar = [y_cutoff_T_1[i] for i in ind_cvar]
    Q_offshore_T_1_cvar = [Q_offshore_T_1[i] for i in ind_cvar]
    cvar_coefficient_f = calculate_cvar_par(alpha, start, end, gas_scenarios, proba_worst_demand, Q_T_1, D_cvar_T_1,
                                            premium,
                                            weather_tot_params, Q_offshore_T_1_cvar, tec, x_cutoff_T_1_cvar,
                                            y_cutoff_T_1_cvar, add_params)  # shape (d,)*(T)

    assert expectation_coefficient_f.shape == cvar_coefficient_f.shape

    expectation_coefficient_f_tilde_beta = []
    for beta in list_beta:  # we calculate expectation_coefficient_f_tilde for all beta values
        expectation_coefficient_f_tilde = beta * expectation_coefficient_f + (1 - beta) * cvar_coefficient_f
        expectation_coefficient_f_tilde = expectation_coefficient_f_tilde.reshape(
            (1,) + expectation_coefficient_f_tilde.shape)
        expectation_coefficient_f_tilde_beta.append(expectation_coefficient_f_tilde)
    expectation_coefficient_f_tilde = np.concatenate(expectation_coefficient_f_tilde_beta, axis=0)

    # TODO: attention !! je teste quelque chose pour la fin de l'horizon, je supprime la dévaluation
    coef_g_values = concatenate_coef_g_no_deval(T, Tprime, Q_T_1, D_average_T_1, premium, weather_params,
                                       Q_offshore_T_1, tec, x_cutoff_T_1, y_cutoff_T_1, add_params)
    # coef_g_values = concatenate_coef_g(T, Tprime, Q_T_1 * (1 - nu_deval), D_average_T_1, premium, weather_params,
    #                                    Q_offshore_T_1, tec, x_cutoff_T_1, y_cutoff_T_1, add_params)

    assert coef_g_values.ndim == T + 1
    # We devaluate one time Q for coef_g_values, as the investment decision was taken at time T-1, and so they have
    # already been devaluated once
    expectation_coefficient_g = (trans_matrix_T_1 * coef_g_values).sum(-1)  # shape (d,)*T

    investment_costs = add_params[f'investment_costs_{tec}']
    # return 1 / (2 * c_tilde) * (
    #         expectation_coefficient_f_tilde - np.exp(-discount_rate) * (1 - nu_deval) * expectation_coefficient_g -
    #         investment_costs[T - 1])  # shape (d,)*T
    # TODO: attention, pareil, je teste quelque chose en enlevant la dévaluation, et je reshape expectation_coefficient_g car il ne prend pas en compte d'aversion au risque
    return 1 / (2 * c_tilde) * (
            expectation_coefficient_f_tilde - np.exp(-discount_rate) * expectation_coefficient_g.reshape((1,) + expectation_coefficient_g.shape) -
            investment_costs[T - 1])  # shape (L,) + (d,)*T
    # We also devaluate one time expectation_coefficient_g for the same reason as previously


def find_optimal_control(T, Tprime, trans_matrix, demand_states, list_beta, alpha, start, end, gas_scenarios, Q: list,
                         premium, weather_params, weather_tot_params, Q_offshore, tec,
                         x_cutoff: list, y_cutoff: list, add_params):
    """
    NEW VERSION. Algorithm computing the optimal control strategy conditioned on other players' action Q
    Parameters
    ----------

    Returns
    -------
    list of length T corresponding to the description of the optimal strategy depending on the observed
    common noise
    Element t of the list is an array of dimension (d,)*(t+1)
    """
    assert len(Q) == T
    # assert type(Q_wind) == type(Q_sun) == np.ndarray
    # assert Q_sun.shape[0] == Q_wind.shape[0] == T  # exogenous strategies must be specified for each possible
    # # time t
    # assert Q_sun.shape[1] == Q_wind.shape[1] == demand_states.shape[0]  # exogenous strategies must be specified at
    # # each time t for each possible demand value
    assert trans_matrix.shape[0] == T  # markov uncertainty demand model must be specified for each time t from 0 to T-1
    assert type(premium) == np.ndarray
    assert len(add_params[f'investment_costs_sun']) == len(add_params[f'investment_costs_wind']) == \
           premium.shape[0] == T
    assert type(x_cutoff) == list, "Parameter x_cutoff should be a list"
    assert type(y_cutoff) == list, "Parameter y_cutoff should be a list"
    assert len(x_cutoff) == len(
        y_cutoff) == T
    # optimal_strategy = np.zeros(Q_sun.shape)
    optimal_strategy = []
    control_final = optimal_control_final(T, Tprime, trans_matrix, demand_states, list_beta, alpha, start, end,
                                          gas_scenarios, Q[T - 1], premium[T - 1], weather_params,
                                          weather_tot_params, Q_offshore[T - 1], tec, x_cutoff[T - 1], y_cutoff[T - 1],
                                          add_params)  # shape (L,) + (d,)*T
    optimal_strategy.insert(0, control_final)
    # optimal_strategy[T - 1, :] = control_final
    control_next = control_final
    for t in range(T - 2, -1, -1):
        control_time_t = recursive_optimal_control(t, control_next, trans_matrix, demand_states, list_beta, alpha, start,
                                                   end, gas_scenarios, Q[t], premium[t], weather_params,
                                                   weather_tot_params, Q_offshore[t], tec, x_cutoff[t], y_cutoff[t],
                                                   add_params)  # shape (L,) + (d,)*(t+1)
        control_next = control_time_t
        # optimal_strategy[t] = control_time_t
        optimal_strategy.insert(0, control_time_t)  # we add new control at the beginning
    # We now apply the thresholding operation
    optimal_strategy_threshold = []
    for t in range(0, T, 1):
        control_time_t_threshold = np.where(optimal_strategy[t] > 0, optimal_strategy[t], 0)
        optimal_strategy_threshold.append(control_time_t_threshold)
    # optimal_strategy = np.where(optimal_strategy > 0, optimal_strategy, 0)  # execute the thresholding operation
    return optimal_strategy_threshold  # should be ok


def state_distribution_from_control(control_sun, control_wind, add_params):
    """Finds state distribution from control for sun and wind technologies
     Returns
    -------
    List of length T where element t corresponds to state distribution at time t
    """
    T = len(control_sun)
    nu_deval = add_params["nu_deval"]
    assert len(control_wind) == T
    state_distribution = []
    state_sun = control_sun[0]  # a modifier si on ajoute une valeur donnée pour l'état initial
    state_wind = control_wind[0]  # a modifier si on ajoute une valeur donnée pour l'état initial
    previous_state = np.array([state_sun, state_wind])  # shape (2,) + (d,)*(t+1) with t=0
    state_distribution.append(previous_state)
    for t in range(1, T, 1):
        state_t_sun = ((1 - nu_deval) * previous_state[0]).reshape(previous_state[0].shape + (1,)) + control_sun[
            t]  # broadcasting to add a final dimension corresponding to new state
        state_t_wind = ((1 - nu_deval) * previous_state[1]).reshape(previous_state[1].shape + (1,)) + control_wind[
            t]
        assert state_t_sun.ndim == t + 1
        assert state_t_wind.ndim == t + 1
        previous_state = np.array([state_t_sun, state_t_wind])  # shape (2,) + (d,)*(t+1)
        state_distribution.append(previous_state)
    return state_distribution


def average_state_distribution_beta(state_distribution, list_weight_beta):
    """Calculates the average state over the whole beta distribution.
    Parameters
    ----------
    state_distribution: list
        Includes a first dimension corresponding to beta heterogeneity
    list_weight_beta: list
        List of weights (proba) associated to each possible value for beta
    """
    new_state_distribution = []
    T = len(state_distribution)
    for t in range(1, T, 1):
        state_sun_t = state_distribution[t][0]
        state_wind_t = state_distribution[t][1]
        average_state_sun_t = (state_sun_t * list_weight_beta.reshape(state_sun_t.shape)).sum(0)  # we average over beta
        average_state_wind_t = (state_wind_t * list_weight_beta.reshape(state_sun_t.shape)).sum(0)  # we average over beta
        new_state_distribution.append(np.array([average_state_sun_t, average_state_wind_t]))
    return new_state_distribution


def control_from_state(state_distribution, add_params):
    """Returns control distribution for sun and wind from the state distribution.
    Returns two lists of length T, where element t is an array of shape (d)*(t+1)"""
    # TODO: peut etre a changer
    T = len(state_distribution)
    nu_deval = add_params["nu_deval"]
    control_distribution_sun = []
    control_distribution_wind = []
    control_sun = state_distribution[0][0]  # a modifier si on ajoute une valeur donnée pour l'état initial
    control_wind = state_distribution[0][1]  # a modifier si on ajoute une valeur donnée pour l'état initial
    # previous_state = state[0]  # shape (2,) + (d,)*(t+1) with t=0
    control_distribution_sun.append(control_sun)
    control_distribution_wind.append(control_wind)
    for t in range(1, T, 1):
        control_t_sun = -((1 - nu_deval) * state_distribution[t - 1][0]).reshape(
            state_distribution[t - 1][0].shape + (1,)) + state_distribution[t][0]  # broadcasting done naturally
        control_t_wind = -((1 - nu_deval) * state_distribution[t - 1][1]).reshape(
            state_distribution[t - 1][1].shape + (1,)) + state_distribution[t][1]  # broadcasting done naturally
        assert control_t_sun.ndim == t + 1
        assert control_t_wind.ndim == t + 1
        control_distribution_sun.append(control_t_sun)
        control_distribution_wind.append(control_t_wind)
    return control_distribution_sun, control_distribution_wind  # should be ok


def decouple_state_distribution(state_distribution):
    """Returns state distribution for sund and wind separated"""
    state_distribution_sun = []
    state_distribution_wind = []
    for i in range(len(state_distribution)):
        state_distribution_sun.append(state_distribution[i][0])
        state_distribution_wind.append(state_distribution[i][1])
    return state_distribution_sun, state_distribution_wind


def cost_time_t(t, opt_control_t, state_distribution_t, other_state_distribution_t, demand_states, proba_time_t,
                trans_matrix, list_beta, alpha, start, end, gas_scenarios, premium, weather_params, weather_tot_params,
                Q_offshore_t, tec, x_cutoff_t, y_cutoff_t, add_params):
    """Returns expected cost at time t specified for each possible uncertainty path"""
    assert opt_control_t.ndim == state_distribution_t.ndim == t + 1 + 1  # here, we added one dimension for the beta
    assert other_state_distribution_t.ndim - 1 == t + 1
    discount_rate, investment_costs, c_tilde = add_params["discount_rate"], add_params[f'investment_costs_{tec}'], \
                                               add_params[f'c_tilde_{tec}']
    trans_matrix_t = trans_matrix[t, :, :]  # shape (d,d)
    proba_time_t_next = proba_time_t.reshape(proba_time_t.shape + (1,)) * \
                        trans_matrix_t.reshape(
                            (1,) * t + trans_matrix_t.shape)  # we calculate the next probability matrix
    # proba_time_t_next has shape (d)*(t+2)
    D_average_t = demand_states[t, :]
    coef_f = concatenate_coef_f(other_state_distribution_t, D_average_t, premium, weather_params, Q_offshore_t,
                                tec, x_cutoff_t, y_cutoff_t, add_params)  # shape (d,)*(t+2)
    assert coef_f.ndim == t + 2  # t+1 dimensions for possible paths until D_{t-1}, and 1 dimension for demand at time t

    trans_matrix_t_cvar = matrix_for_cvar(trans_matrix_t)
    D_cvar_t = (D_average_t * trans_matrix_t_cvar).sum(-1)  # shape (d,)*(t+1)
    proba_worst_demand = probability_worst_demand(trans_matrix_t)  # shape (d,)
    ind_cvar = index_cvar(trans_matrix_t)
    x_cutoff_t_cvar = [x_cutoff_t[i] for i in ind_cvar]
    y_cutoff_t_cvar = [y_cutoff_t[i] for i in ind_cvar]
    Q_offshore_t_cvar = [Q_offshore_t[i] for i in ind_cvar]
    # TODO: je change cette ligne pour tester le parallélisme
    cvar_coefficient_f = calculate_cvar_par(alpha, start, end, gas_scenarios, proba_worst_demand,
                                            other_state_distribution_t, D_cvar_t, premium, weather_tot_params,
                                            Q_offshore_t_cvar, tec, x_cutoff_t_cvar, y_cutoff_t_cvar,
                                            add_params)  # shape (d,)*(t+1)
    assert state_distribution_t[0,...].shape == cvar_coefficient_f.shape == proba_time_t.shape  # we need to remove first dimension associated with beta

    list_cost_beta = []
    for i in range(list_beta):  # here, we pay attention to selecting only state and control associated with given beta
        cvar_cost_beta = -(1 - list_beta[i]) * np.exp(-discount_rate * t) * (
                state_distribution_t[i, ...] * cvar_coefficient_f * proba_time_t)  # shape (d,)*(t+1)
        investment_cost_beta = np.exp(-discount_rate * t) * (
                investment_costs[t] * opt_control_t[i, ...] + c_tilde * opt_control_t[i, ...] ** 2) * proba_time_t  # shape  (d,)*(t+1)
        expected_cost_beta = - list_beta[i] * np.exp(-discount_rate * t) * (state_distribution_t[i,...].reshape(
            state_distribution_t[i,...].shape + (1,)) * coef_f * proba_time_t_next)  # shape (d,)*(t+2)
        # remark: we reshape to add a dimension for the beta coefficient
        # TODO: il y a peut-être des erreurs de broadcast ici

        cost_beta = cvar_cost_beta.sum() + investment_cost_beta.sum() + expected_cost_beta.sum()  # float
        list_cost_beta.append(cost_beta)
    return list_cost_beta, proba_time_t_next


def player_cost(T, Tprime, opt_control, state_distribution, other_state_distribution, trans_matrix, demand_states, list_beta,
                list_weight_beta, alpha, start, end, gas_scenarios, premium, weather_params, weather_tot_params, Q_offshore, tec,
                x_cutoff: list, y_cutoff: list, add_params):
    """Returns total player's cost for a given strategy and corresponding state distribution"""
    discount_rate, nu_deval = add_params["discount_rate"], add_params["nu_deval"]
    assert len(opt_control) == len(state_distribution) == len(other_state_distribution) == trans_matrix.shape[0] == T
    proba_time_t = np.array([0, 1, 0])  # we initialize the probability by stating that we are at state 1 when t=-1
    total_cost = 0
    list_total_cost_beta = [0 for i in range(list_beta)]  # initialize costs to zero
    for t in range(0, T, 1):
        opt_control_t = opt_control[t]
        if tec == 'sun':
            state_distribution_t = state_distribution[t][0]  # includes beta heterogeneity
        else:  # tec == 'wind'
            state_distribution_t = state_distribution[t][1]  # includes beta heterogeneity
        other_state_distribution_t = other_state_distribution[t]  # already integrated over the beta heterogeneity
        assert proba_time_t.ndim == t + 1
        list_cost_beta_t, proba_time_t_next = cost_time_t(t, opt_control_t, state_distribution_t, other_state_distribution_t,
                                                demand_states, proba_time_t, trans_matrix, list_beta, alpha, start, end,
                                                          gas_scenarios, premium[t], weather_params, weather_tot_params,
                                                          Q_offshore[t], tec, x_cutoff[t], y_cutoff[t], add_params)
        list_total_cost_beta = [total_cost + cost_t for (total_cost, cost_t) in zip(list_total_cost_beta, list_cost_beta_t)]  # TODO: il y a peut-être un problème ici
        # total_cost = total_cost + cost_t
        proba_time_t = proba_time_t_next  # shape (d,)*(t+2)
    D_average_T = demand_states[T - 1, :]
    # Q_T = other_state_distribution[T - 1] * (1 - nu_deval)
    Q_T = other_state_distribution[T - 1]  # TODO: attention, j'ai changé cela pour arrêter la dévaluation, il pourrait y avoir une erreur
    coef_g = concatenate_coef_g_no_deval(T, Tprime, Q_T, D_average_T, premium[T - 1], weather_params,
                                Q_offshore[T - 1], tec, x_cutoff[T - 1], y_cutoff[T - 1],
                                add_params)  # shape (d,)*(T+1)

    # if tec == 'sun':
    #     cost_final = np.exp(-discount_rate * T) * \
    #                  (state_distribution[T - 1][0] * (1 - nu_deval)).reshape(
    #                      state_distribution[T - 1][0].shape + (1,)) * coef_g  # shape (d,)*(T+1)
    # else:
    #     cost_final = np.exp(-discount_rate * T) * \
    #                  (state_distribution[T - 1][1] * (1 - nu_deval)).reshape(
    #                      state_distribution[T - 1][1].shape + (1,)) * coef_g  # shape (d,)*(T+1)
    list_final_cost_beta = []
    if tec == 'sun':  # TODO: pareil, j'ai changé pour enlever la dévaluation de coef_g, il pourrait y avoir des bugs
        for i in range(list_beta):
            cost_final_beta = np.exp(-discount_rate * T) * \
                         (state_distribution[T - 1][0][i,...]).reshape(
                             state_distribution[T - 1][0][i,...].shape + (1,)) * coef_g  # shape (d,)*(T+1)
            assert cost_final_beta.shape == proba_time_t.shape
            cost_final_beta_with_proba = cost_final_beta * proba_time_t
            cost_T_beta = cost_final_beta_with_proba.sum()
            list_final_cost_beta.append(cost_T_beta)
    else:
        for i in range(list_beta):
            cost_final_beta = np.exp(-discount_rate * T) * \
                         (state_distribution[T - 1][1][i,...]).reshape(
                             state_distribution[T - 1][1][i,...].shape + (1,)) * coef_g  # shape (d,)*(T+1)
            assert cost_final_beta.shape == proba_time_t.shape
            cost_final_beta_with_proba = cost_final_beta * proba_time_t
            cost_T_beta = cost_final_beta_with_proba.sum()
            list_final_cost_beta.append(cost_T_beta)

    list_total_cost_beta = [total_cost + final_cost for (total_cost, final_cost) in zip(list_total_cost_beta, list_final_cost_beta)]
    # total_cost += cost_T
    total_cost = sum([total_cost * weight_beta for (total_cost, weight_beta) in zip(list_total_cost_beta, list_weight_beta)])  # here, we take the average over all values for beta
    return total_cost


def fictitious_play(N, T, Tprime, state_init: list, trans_matrix, demand_states, list_beta, list_weight_beta, alpha, start, end,
                    gas_scenarios, premium, weather_params, weather_tot_params, Q_offshore, x_cutoff, y_cutoff,
                    add_params, convergence):

    """

    Parameters
    ----------
    N
    T
    Tprime
    state_init: list
        Element t is of dim (2,) + (d,)*(t+1) and corresponds to state distribution for sun and wind
    Returns
    -------

    """
    assert len(
        state_init) == T, f"Initial state distribution for other players should be of length {T}, instead length " \
                          f"{len(state_init)} "
    assert convergence in ['wolfe', 'fictitious'], "Parameter convergence does not take allowed value."
    average_state_distribution = state_init
    state_distribution = []  # we create an initial distribution including beta heterogeneity
    for t in range(0, len(average_state_distribution), 1):
        state_distribution_sun_t = np.repeat(average_state_distribution[t][0][np.newaxis, ...], len(list_beta), axis=0)
        state_distribution_wind_t = np.repeat(average_state_distribution[t][1][np.newaxis, ...], len(list_beta), axis=0)
        state_distribution.append(np.array([state_distribution_sun_t, state_distribution_wind_t]))

    previous_control_sun, previous_control_wind = control_from_state(state_distribution, add_params)  # includes beta heterogeneity
    history_state_distribution = []  # maybe necessary if we want to check if optimal control moves a lot or not
    objective_gap = []
    index_objective_gap = []
    history_state_distribution.append(average_state_distribution)

    for n in range(N):
        print(n, flush=True)
        TIKTOK.reset()

        # Finding optimal control
        TIKTOK.interval("optimal_control", display=False)
        opt_control_sun = find_optimal_control(T, Tprime, trans_matrix, demand_states, list_beta, alpha, start, end,
                                               gas_scenarios, average_state_distribution, premium, weather_params,
                                               weather_tot_params, Q_offshore, "sun", x_cutoff, y_cutoff, add_params)

        opt_control_wind = find_optimal_control(T, Tprime, trans_matrix, demand_states, list_beta, alpha, start, end,
                                                gas_scenarios, average_state_distribution, premium, weather_params,
                                                weather_tot_params, Q_offshore, "wind", x_cutoff, y_cutoff, add_params)

        # Finding associated state distribution
        new_state_distribution = state_distribution_from_control(opt_control_sun, opt_control_wind, add_params)  # includes the different beta
        new_average_state_distribution = average_state_distribution_beta(new_state_distribution, list_weight_beta)  # averaged over the beta

        if n % 20 == 0:
            TIKTOK.interval("optimal_control", display=True)
            # TODO: a corriger !! il faut modifier l'appel à new_state_distribution, car on a des états qui diffèrent selon le paramètre beta
            current_player_cost_sun = player_cost(T, Tprime, opt_control_sun, new_state_distribution,
                                                  average_state_distribution, trans_matrix, demand_states, list_beta, list_weight_beta,
                                                  alpha, start, end, gas_scenarios, premium, weather_params, weather_tot_params,
                                                  Q_offshore, "sun", x_cutoff, y_cutoff, add_params)

            previous_player_cost_sun = player_cost(T, Tprime, previous_control_sun, state_distribution,
                                                   average_state_distribution, trans_matrix, demand_states, list_beta, list_weight_beta,
                                                   alpha, start, end, gas_scenarios, premium, weather_params, weather_tot_params,
                                                   Q_offshore, "sun", x_cutoff, y_cutoff, add_params)

            current_player_cost_wind = player_cost(T, Tprime, opt_control_wind, new_state_distribution,
                                                   average_state_distribution, trans_matrix, demand_states, list_beta, list_weight_beta,
                                                   alpha, start, end, gas_scenarios, premium, weather_params, weather_tot_params,
                                                   Q_offshore, "wind", x_cutoff, y_cutoff, add_params)

            previous_player_cost_wind = player_cost(T, Tprime, previous_control_wind, state_distribution,
                                                    average_state_distribution, trans_matrix, demand_states, list_beta, list_weight_beta,
                                                    alpha, start, end, gas_scenarios, premium, weather_params, weather_tot_params,
                                                    Q_offshore, "wind", x_cutoff, y_cutoff, add_params)

            gap = (current_player_cost_sun - previous_player_cost_sun) + (
                    current_player_cost_wind - previous_player_cost_wind)

            # TODO: check that the gap remains negative
            objective_gap.append(gap)
            index_objective_gap.append(n + 1)
            if n % 100 == 0:
                print(f'Objective gap: {objective_gap}', flush=True)
        if convergence == 'wolfe':
            average_state_distribution = [2 / (n + 2) * new_state + n / (n + 2) * state
                                  for new_state, state in zip(new_average_state_distribution, average_state_distribution)]
            state_distribution = [2 / (n + 2) * new_state + n / (n + 2) * state
                                          for new_state, state in zip(new_state_distribution, state_distribution)]
        else:  # 'fictitious'
            average_state_distribution = [1 / (n + 1) * new_state + n / (n + 1) * state
                                  for new_state, state in zip(new_average_state_distribution, average_state_distribution)]
            state_distribution = [1 / (n + 1) * new_state + n / (n + 1) * state
                                  for new_state, state in zip(new_state_distribution, state_distribution)]
        previous_control_sun, previous_control_wind = control_from_state(state_distribution, add_params)
    return state_distribution, np.array(objective_gap), np.array(index_objective_gap)

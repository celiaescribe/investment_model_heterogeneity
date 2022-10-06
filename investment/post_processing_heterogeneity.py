import numpy as np
import json
from matplotlib import pyplot as plt

from investment.data_loading import data_process_with_scenarios
from investment.demand_model import demand_model
from investment.optimization_model import control_from_state, decouple_state_distribution, supply_function_np_piecewise, \
    state_distribution_from_control
from datetime import datetime
from pathlib import Path
import os
import seaborn as sns
import pandas as pd
import plotly.express as px


def state_from_demand_trajectory_heterogeneity(list_beta, time_list, demand_trajectory, state_distribution):
    """Returns two lists of sequential state values for technology sun and wind, corresponding to a
    given demand trajectory, where the demand trajectory is specified from time -1 to time T-2 (as state at time
    T-1 depends only on demand until time T-2."""
    state_trajectory_sun, state_trajectory_wind = pd.DataFrame({"time": pd.Series(dtype='str'),
                                                                "beta": pd.Series(dtype="float"),
                                                                "state": pd.Series(dtype="float")}), \
                                                    pd.DataFrame({"time": pd.Series(dtype='str'),
                                                                  "beta": pd.Series(dtype="float"),
                                                                  "state": pd.Series(dtype="float")})
    for t in range(len(state_distribution)):
        time = time_list[t]
        state_sun_t = state_distribution[t][0]  # sun distribution at time t
        state_wind_t = state_distribution[t][1]  # wind distribution at time t
        for i in range(
                t + 1):  # state at time t depends from demand value from t=-1 to t-1 (which is a total of t+1 values)
            state_sun_t = state_sun_t[:, demand_trajectory[i]]  # we keep the initial dimension corresponding to beta
            state_wind_t = state_wind_t[:, demand_trajectory[i]]  # we keep the initial dimension corresponding to beta
        state_dataframe_sun_t = pd.DataFrame({'time': [time]*len(list_beta), 'beta': list_beta, 'state': state_sun_t})
        state_dataframe_wind_t = pd.DataFrame({'time': [time] * len(list_beta), 'beta': list_beta, 'state': state_sun_t})
        state_trajectory_sun = pd.concat([state_trajectory_sun, state_dataframe_sun_t], ignore_index=True)
        state_trajectory_wind = pd.concat([state_trajectory_wind, state_dataframe_wind_t], ignore_index=True)
    return state_trajectory_sun, state_trajectory_wind


# def control_from_demand_trajectory_heterogeneity(state_trajectory_sun, state_trajectory_wind, add_params):
#     nu_deval = add_params["nu_deval"]
#     state_trajectory_sun["state_deval"] = state_trajectory_sun["state"]*(1-nu_deval)
#     state_trajectory_wind["state_deval"] = state_trajectory_wind["state"] * (1 - nu_deval)
#     state_trajectory_sun["control"] = 0
#     return 0


def create_state_dataframe_2param(state_distribution_dict, demand_trajectory_dict, demand_list, keyword1: str,
                                  keyword2: str, param_list, time_list, list_beta):
    """Returns a dataframe compiling state values for different model parameters for two parameters
    (risk aversion, ...), for different technologies, and different demand trajectories."""
    state_dataframe = pd.DataFrame(
        {"time": pd.Series(dtype='str'), "state": pd.Series(dtype="float"), "tec": pd.Series(dtype="str"),
         keyword1: pd.Series(dtype="str"), keyword2: pd.Series(dtype="str"), "demand_trajectory": pd.Series(dtype="str")})

    for param in param_list:
        (param1, param2) = param
        # print(f"Param1 : {param1}, Param2: {param2}")
        for demand_traj in demand_list:
            state_sun, state_wind = state_from_demand_trajectory_heterogeneity(list_beta, time_list,
                                                                               demand_trajectory_dict[demand_traj],
                                                                               state_distribution_dict[param])
            state_sun["tec"], state_sun[keyword1] = "sun", param1  # a verifier, peut etre pas le bon code pour observer ce qu'on veut
            state_wind["tec"], state_wind[keyword1] = "wind", param1
            # state = pd.merge(state_sun, state_wind, on=["time", "beta"])
            #  state = state.rename(columns)   # il faut renommer les colonnes !

            new_df = pd.DataFrame(
                {'time': time_list, 'state': state_sun, 'control': control_sun, 'tec': ["sun"] * len(time_list),
                 keyword1: [param1] * len(time_list), keyword2: [param2] * len(time_list),
                 'demand_trajectory': [demand_traj] * len(time_list)})
            state_dataframe = pd.concat([state_dataframe, new_df], ignore_index=True)
            new_df = pd.DataFrame(
                {'time': time_list, 'state': state_wind, 'control': control_wind, 'tec': ["wind"] * len(time_list),
                 keyword1: [param1] * len(time_list), keyword2: [param2] * len(time_list),
                 'demand_trajectory': [demand_traj] * len(time_list)})
            state_dataframe = pd.concat([state_dataframe, new_df], ignore_index=True)
    return state_dataframe
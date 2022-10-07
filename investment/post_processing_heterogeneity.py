import numpy as np
import json
from matplotlib import pyplot as plt

from investment.data_loading import data_process_with_scenarios
from investment.demand_model import demand_model
from investment.optimization_model import control_from_state, decouple_state_distribution, supply_function_np_piecewise, \
    state_distribution_from_control
from investment.post_processing import filter_dirs
from datetime import datetime
from pathlib import Path
import os
import pandas as pd


def load_results_heterogeneity(path):
    """Load results, including description of beta distribution."""
    control_sun = np.load(f'{path}_control_sun.npz')
    control_sun_unzip = []
    control_wind = np.load(f'{path}_control_wind.npz')
    control_wind_unzip = []
    for t in control_sun.files:
        control_sun_unzip.append(control_sun[t])
    for t in control_wind.files:
        control_wind_unzip.append(control_wind[t])
    objective = np.load(f'{path}_objective.npz')
    objective_gap = objective['objective']
    index_objective_gap = objective['index']
    beta = np.load(f'{path}_beta_discretization.npz')
    list_beta = beta['list_beta']
    list_weight_beta = beta['list_weight_beta']
    with open(f'{path}_additional_parameters.json', 'r') as json_file:
        additional_parameters = json.load(json_file)
    return control_sun_unzip, control_wind_unzip, objective_gap, index_objective_gap, list_beta, list_weight_beta, additional_parameters


def process_files_from_folder_heterogeneity(directory, keyword1, keyword2):
    """Process all folders in a given folder to extract information"""
    state_distribution_dict = {}
    objective_gap_dict = {}
    index_objective_dict = {}
    list_beta_dict = {}
    list_weight_beta_dict = {}

    L1 = len(keyword1)
    L2 = len(keyword2)

    list_paths = filter_dirs(directory)
    for path in list_paths:
        print(path)
        hour = (filter_dirs(directory + path)[0]).split('_')[0]  # get hour at which files were written
        for s in path.split('_'):
            if keyword1 in s:
                param1 = s[L1:]
            if keyword2 in s:
                param2 = s[L2:]

        total_path = directory + path + '/' + hour
        control_sun, control_wind, objective_gap, index_objective_gap, list_beta, list_weight_beta, additional_parameters = load_results_heterogeneity(total_path)
        state_distribution = state_distribution_from_control(control_sun, control_wind, additional_parameters)
        state_distribution_dict[(param1, param2)] = state_distribution
        objective_gap_dict[(param1, param2)] = objective_gap
        index_objective_dict[(param1, param2)] = index_objective_gap
        list_beta_dict[(param1, param2)] = list_beta
        list_weight_beta_dict[(param1, param2)] = list_weight_beta
    return state_distribution_dict, objective_gap_dict, index_objective_dict, list_beta_dict, list_weight_beta_dict


def state_from_demand_trajectory_heterogeneity(list_beta, list_weight_beta, time_list, demand_trajectory, state_distribution):
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
        state_dataframe_sun_t = pd.DataFrame({'time': [time]*len(list_beta), 'beta': list_beta, 'beta_weight': list_weight_beta, 'state': state_sun_t})
        state_dataframe_wind_t = pd.DataFrame({'time': [time] * len(list_beta), 'beta': list_beta, 'beta_weight': list_weight_beta, 'state': state_wind_t})
        state_trajectory_sun = pd.concat([state_trajectory_sun, state_dataframe_sun_t], ignore_index=True)
        state_trajectory_wind = pd.concat([state_trajectory_wind, state_dataframe_wind_t], ignore_index=True)
    return state_trajectory_sun, state_trajectory_wind


# def control_from_demand_trajectory_heterogeneity(state_trajectory_sun, state_trajectory_wind, add_params):
#     nu_deval = add_params["nu_deval"]
#     state_trajectory_sun["state_deval"] = state_trajectory_sun["state"]*(1-nu_deval)
#     state_trajectory_wind["state_deval"] = state_trajectory_wind["state"] * (1 - nu_deval)
#     state_trajectory_sun["control"] = 0
#     return 0


def create_state_dataframe_heterogeneity(state_distribution_dict, list_beta_dict, list_weight_beta_dict,
                                         demand_trajectory_dict, demand_list, keyword1: str,
                                         keyword2: str, param_list, time_list):
    """Returns a dataframe compiling state values for different model parameters for two parameters
    (risk aversion, ...), for different technologies, and different demand trajectories."""
    state_dataframe = pd.DataFrame(
        {"time": pd.Series(dtype='str'), "state": pd.Series(dtype="float"), "tec": pd.Series(dtype="str"),
         keyword1: pd.Series(dtype="str"), keyword2: pd.Series(dtype="str"), "demand_trajectory": pd.Series(dtype="str"),
         "beta": pd.Series(dtype='float'), 'beta_weight': pd.Series(dtype='float')})

    for param in param_list:
        (param1, param2) = param
        # print(f"Param1 : {param1}, Param2: {param2}")
        for demand_traj in demand_list:
            state_sun, state_wind = state_from_demand_trajectory_heterogeneity(list_beta_dict[param],
                                                                               list_weight_beta_dict[param], time_list,
                                                                               demand_trajectory_dict[demand_traj],
                                                                               state_distribution_dict[param])
            state_sun["tec"], state_sun[keyword1], state_sun[keyword2], state_sun["demand_trajectory"] = "sun", param1, param2, demand_traj  # a verifier, peut etre pas le bon code pour observer ce qu'on veut
            state_wind["tec"], state_wind[keyword1], state_wind[keyword2], state_wind["demand_trajectory"] = "wind", param1, param2, demand_traj
            # state = pd.merge(state_sun, state_wind, on=["time", "beta"])
            #  state = state.rename(columns)   # il faut renommer les colonnes !
            state_dataframe = pd.concat([state_dataframe, state_sun], ignore_index=True)
            state_dataframe = pd.concat([state_dataframe, state_wind], ignore_index=True)
    return state_dataframe

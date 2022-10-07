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
# import seaborn as sns
import pandas as pd
# import plotly.express as px


def process_output(T, state_distribution, objective_gap, index_objective_gap, list_beta, list_weight_beta, subdirectory_name, add_params):
    """Save output"""
    # Process output
    control_sun, control_wind = control_from_state(state_distribution, add_params)
    state_sun, state_wind = decouple_state_distribution(state_distribution)
    control = {'sun': control_sun, 'wind': control_wind}
    state = {'sun': state_sun, 'wind': state_wind}

    list_cluster = ['sun', 'wind']
    for cluster in list_cluster:
        control_by_time = {}
        for t in range(T):
            control_by_time[f't{t}'] = control[cluster][t]
        control[cluster] = control_by_time

    # Save control values to file
    filename_hour = datetime.now().strftime("%H-%M-%S")
    for cluster in list_cluster:
        path = subdirectory_name + f"/{filename_hour}_control_{cluster}.npz"
        np.savez(path, **control[cluster])

    # Save objective values to file
    np.savez(subdirectory_name + f"/{filename_hour}_objective.npz",
             **{'objective': objective_gap, 'index': index_objective_gap})

    # Save beta discretization to file
    np.savez(subdirectory_name + f"/{filename_hour}_beta_discretization.npz",
             **{'list_beta': list_beta, 'list_weight_beta': list_weight_beta})

    # save json config
    with open(Path(subdirectory_name) / f'{filename_hour}_additional_parameters.json', 'a') as fp:
        json.dump(add_params, fp, indent=4)


def load_results(path):
    """Load results"""
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
    with open(f'{path}_additional_parameters.json', 'r') as json_file:
        additional_parameters = json.load(json_file)
    return control_sun_unzip, control_wind_unzip, objective_gap, index_objective_gap, additional_parameters


def filter_dirs(directory):
    """Extracts all files from a directory"""
    files = os.listdir(directory)
    files_extracted = []
    for file in files:
        if file != '.DS_Store':
            files_extracted.append(file)
    return sorted(files_extracted)


def load_classical_params():
    """Returns CLASSICAL set of params. Attention, any modification in the parameters may require to also modify the
    results returned by this function. Notably market price cap, transition matrix, and availability of nuclear."""
    T = 5
    Tprime = 8
    time_step = 5  # number of years between investment decisions
    discount_rate = time_step * 0.04
    nu_deval = 0.15
    market_price = 10000  # market price cap, MAY NEED TO BE MODIFIED
    Delta_t = time_step * 365 * 24  # number of hours in ergodic theorem
    voll = 15000  # value of loss load
    c_tilde_sun = 60000  # sun, wind
    c_tilde_wind = 120000
    investment_costs_sun = [6 * 1e5, 5.5 * 1e5, 5.2 * 1e5, 5 * 1e5, 5 * 1e5]
    investment_costs_wind = [12.5 * 1e5, 12 * 1e5, 11.1 * 1e5, 10.3 * 1e5, 9.8 * 1e5]
    Q_pv_init = 10
    Q_onshore_init = 18
    Q_river = 10

    additional_parameters = {'market_price': market_price,
                             'voll': voll,
                             'discount_rate': discount_rate,
                             'nu_deval': nu_deval,
                             'Delta_t': Delta_t,
                             'c_tilde_sun': c_tilde_sun,
                             'c_tilde_wind': c_tilde_wind,
                             'Q_pv_init': Q_pv_init,
                             'Q_onshore_init': Q_onshore_init,
                             'Q_river': Q_river,
                             'investment_costs_sun': investment_costs_sun,
                             'investment_costs_wind': investment_costs_wind}

    # Demand scenarios
    trans_matrix, demand_states = demand_model()

    # Joint law
    joint_law, x_cutoff, y_cutoff, Q_offshore = data_process_with_scenarios(period="large",
                                                                            avail_nuclear=1)  # ATTENTION, avail_nuclear may need to be modified
    demand_centered = np.array(joint_law.demand_centered_group)
    pv = np.array(joint_law.pv_group)
    onshore = np.array(joint_law.onshore_group)
    offshore = np.array(joint_law.offshore_group)
    river = np.array(joint_law.river_group)
    hydro_prod = np.array(joint_law.hydro_prod_group)
    probability = np.array(joint_law.probability)
    weather_params = {
        "demand": demand_centered,
        "sun": pv,
        "onshore": onshore,
        "offshore": offshore,
        "river": river,
        "hydro": hydro_prod,
        "proba": probability
    }

    return additional_parameters, trans_matrix, demand_states, joint_law, x_cutoff, y_cutoff, Q_offshore, weather_params


def plot_objective(index_gap, objective_gap):
    """Plots objective to check convergence"""
    plt.plot(np.log(index_gap), np.log(-objective_gap))
    plt.show()


def control_from_demand_trajectory(demand_trajectory, control_sun, control_wind):
    """Compiles control for a given demand trajectory"""
    assert len(control_sun) == len(control_wind), "Control lists do not have same length"
    control_trajectory_sun, control_trajectory_wind = [], []
    for t in range(len(control_sun)):
        control_sun_t = control_sun[t]
        control_wind_t = control_wind[t]
        for i in range(t + 1):
            control_sun_t = control_sun_t[demand_trajectory[i]]
            control_wind_t = control_wind_t[demand_trajectory[i]]
        control_trajectory_sun.append(control_sun_t)
        control_trajectory_wind.append(control_wind_t)
    return control_trajectory_sun, control_trajectory_wind


def process_files_from_folder(directory, keyword):
    """Process all folders in a given folder to extract information"""
    state_distribution_dict = {}
    objective_gap_dict = {}
    index_objective_dict = {}

    L = len(keyword)

    list_paths = filter_dirs(directory)
    for path in list_paths:
        print(path)
        hour = (filter_dirs(directory + path)[0]).split('_')[0]  # get hour at which files were written
        param = path.split('_')[0][L:]  # get value of param for each of the files
        total_path = directory + path + '/' + hour
        control_sun, control_wind, objective_gap, index_objective_gap, additional_parameters = load_results(total_path)
        state_distribution = state_distribution_from_control(control_sun, control_wind, additional_parameters)
        state_distribution_dict[param] = state_distribution
        objective_gap_dict[param] = objective_gap
        index_objective_dict[param] = index_objective_gap
    return state_distribution_dict, objective_gap_dict, index_objective_dict


def process_files_from_folder_2kwd(directory, keyword1, keyword2):
    """Process all folders in a given folder to extract information"""
    state_distribution_dict = {}
    objective_gap_dict = {}
    index_objective_dict = {}

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
        control_sun, control_wind, objective_gap, index_objective_gap, additional_parameters = load_results(total_path)
        state_distribution = state_distribution_from_control(control_sun, control_wind, additional_parameters)
        state_distribution_dict[(param1, param2)] = state_distribution
        objective_gap_dict[(param1, param2)] = objective_gap
        index_objective_dict[(param1, param2)] = index_objective_gap
    return state_distribution_dict, objective_gap_dict, index_objective_dict


def process_files_from_folder_3kwd(directory, keyword1, keyword2, keyword3):
    """Process all folders in a given folder to extract information"""
    state_distribution_dict = {}
    objective_gap_dict = {}
    index_objective_dict = {}

    L1 = len(keyword1)
    L2 = len(keyword2)
    L3 = len(keyword3)

    list_paths = filter_dirs(directory)
    for path in list_paths:
        print(path)
        hour = (filter_dirs(directory + path)[0]).split('_')[0]  # get hour at which files were written
        for s in path.split('_'):
            if keyword1 in s:
                param1 = s[L1:]
            if keyword2 in s:
                param2 = s[L2:]
            if keyword3 in s:
                param3 = s[L3:]

        total_path = directory + path + '/' + hour
        control_sun, control_wind, objective_gap, index_objective_gap, additional_parameters = load_results(total_path)
        state_distribution = state_distribution_from_control(control_sun, control_wind, additional_parameters)
        state_distribution_dict[(param1, param2, param3)] = state_distribution
        objective_gap_dict[(param1, param2, param3)] = objective_gap
        index_objective_dict[(param1, param2, param3)] = index_objective_gap
    return state_distribution_dict, objective_gap_dict, index_objective_dict


def convergence_rate_2param(state_distribution_dict, index_objective_dict, objective_gap_dict, param_list,
                            keyword1: str, keyword2: str):
    """Dataframe comparing the convergence rate for the objective"""
    convergence_dataframe = pd.DataFrame(columns=["index_objective", "objective_gap", "convergence"])
    for param in param_list:
        (param1, param2) = param
        new_df = pd.DataFrame({'index_objective': index_objective_dict[param],
                               'objective_gap': objective_gap_dict[param],
                               keyword1: [param1] * index_objective_dict[param].shape[0],
                               keyword2: [param2] * index_objective_dict[param].shape[0]}
                              )
        convergence_dataframe = pd.concat([convergence_dataframe, new_df], ignore_index=True)
    return convergence_dataframe


def plot_convergence_rate_2param(convergence_dataframe, keyword1, keyword2, show=True):
    """Plot evolution of convergence rate"""
    p = px.line(convergence_dataframe, x="index_objective", y="objective_gap", color=keyword1, line_dash=keyword2,
                title="Evolution of objective gap", labels={
            "index_objective": "Iteration",
            "objective_gap": "Objective gap"
        }, )
    if show:
        p.show()
    return p


def state_from_demand_trajectory(demand_trajectory, state_distribution):
    """Returns two lists of sequential state values for technology sun and wind, corresponding to a
    given demand trajectory, where the demand trajectory is specified from time -1 to time T-2 (as state at time
    T-1 depends only on demand until time T-2."""
    state_trajectory_sun, state_trajectory_wind = [], []
    for t in range(len(state_distribution)):
        state_sun_t = state_distribution[t][0]  # sun distribution at time t
        state_wind_t = state_distribution[t][1]  # wind distribution at time t
        for i in range(
                t + 1):  # state at time t depends from demand value from t=-1 to t-1 (which is a total of t+1 values)
            state_sun_t = state_sun_t[demand_trajectory[i]]
            state_wind_t = state_wind_t[demand_trajectory[i]]
        state_trajectory_sun.append(state_sun_t)
        state_trajectory_wind.append(state_wind_t)
    return state_trajectory_sun, state_trajectory_wind


def control_from_state_trajectory(state_trajectory_sun, state_trajectory_wind, add_params):
    """Calculates the invested quantity at time t based on state at time t, for all t for a given demand trajectory"""
    nu_deval = add_params["nu_deval"]
    control_trajectory_sun, control_trajectory_wind = [state_trajectory_sun[0]], [state_trajectory_wind[0]]
    for t in range(1, len(state_trajectory_sun)):
        control_t_sun = -(1 - nu_deval) * state_trajectory_sun[t - 1] + state_trajectory_sun[t]
        control_t_wind = -(1 - nu_deval) * state_trajectory_wind[t - 1] + state_trajectory_wind[t]
        control_trajectory_sun.append(control_t_sun)
        control_trajectory_wind.append(control_t_wind)
    return control_trajectory_sun, control_trajectory_wind


def create_state_dataframe(key: str, state_distribution_dict, demand_trajectory_dict, time_list, add_params):
    """Returns a dataframe compiling state values for different model parameters for one parameter only
    (risk aversion, ...), for different technologies, and different demand trajectories."""
    state_dataframe = pd.DataFrame(columns=["time", "state", "control", "tec", key, "demand_trajectory"])
    for parameter in list(state_distribution_dict.keys()):
        for demand_traj in ["low", "middle", "high"]:
            state_sun, state_wind = state_from_demand_trajectory(demand_trajectory_dict[demand_traj],
                                                                 state_distribution_dict[parameter])
            control_sun, control_wind = control_from_state_trajectory(state_sun, state_wind, add_params)
            new_df = pd.DataFrame(
                {'time': time_list, 'state': state_sun, 'control': control_sun, 'tec': ["sun"] * len(time_list),
                 key: [parameter] * len(time_list),
                 'demand_trajectory': [demand_traj] * len(time_list)})
            state_dataframe = pd.concat([state_dataframe, new_df], ignore_index=True)
            new_df = pd.DataFrame(
                {'time': time_list, 'state': state_wind, 'control': control_wind, 'tec': ["wind"] * len(time_list),
                 key: [parameter] * len(time_list),
                 'demand_trajectory': [demand_traj] * len(time_list)})
            state_dataframe = pd.concat([state_dataframe, new_df], ignore_index=True)
    return state_dataframe


def create_state_dataframe_2param(state_distribution_dict, demand_trajectory_dict, demand_list, keyword1: str,
                                  keyword2: str, param_list, time_list, add_params):
    """Returns a dataframe compiling state values for different model parameters for two parameters
    (risk aversion, ...), for different technologies, and different demand trajectories."""
    state_dataframe = pd.DataFrame(
        {"time": pd.Series(dtype='str'), "state": pd.Series(dtype="float"), "control": pd.Series(dtype="float"),
         "tec": pd.Series(dtype="str"),
         keyword1: pd.Series(dtype="str"), keyword2: pd.Series(dtype="str"),
         "demand_trajectory": pd.Series(dtype="str")})

    for param in param_list:
        (param1, param2) = param
        # print(f"Param1 : {param1}, Param2: {param2}")
        for demand_traj in demand_list:
            state_sun, state_wind = state_from_demand_trajectory(demand_trajectory_dict[demand_traj],
                                                                 state_distribution_dict[param])
            control_sun, control_wind = control_from_state_trajectory(state_sun, state_wind, add_params)
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


def plot_state_trajectory_curve(state_dataframe, static=True, keyword1="", show=True):
    """Plot state trajectory curves for different demand trajectories"""
    if static:
        p = sns.lineplot(x="time", y="state", hue="demand_trajectory", style=keyword1,
                         data=state_dataframe, legend=True)
        p.set_title("Evolution of total installed capacities")
        p.set_xlabel("Time", fontsize=10)
        p.set_ylabel("Total capacity installed (GW)", fontsize=10)
        if show:
            plt.show()
    else:
        p = px.line(state_dataframe, x="time", y="state", color="demand_trajectory", line_dash=keyword1,
                    title="Evolution of total installed capacities", labels={
                "time": "Time",
                "state": "Total capacity installed (GW)"
            }, )
        if show:
            p.show()
    return p


def plot_state_trajectory_curve_2param(state_dataframe, static=True, keyword1="", keyword2="", show=True):
    """Plot state trajectory curves for different demand trajectories"""
    # TODO: il faut ajouter keyword2 dans la version statique
    if static:
        p = sns.lineplot(x="time", y="state", hue="demand_trajectory", style=keyword1,
                         data=state_dataframe, legend=True)
        p.set_title("Evolution of total installed capacities")
        p.set_xlabel("Time", fontsize=10)
        p.set_ylabel("Total capacity installed (GW)", fontsize=10)
        if show:
            plt.show()
    else:
        p = px.line(state_dataframe, x="time", y="state", color="demand_trajectory", line_dash=keyword1,
                    facet_col=keyword2,
                    title="Evolution of total installed capacities", labels={
                "time": "Time",
                "state": "Total capacity installed (GW)"
            }, )
        if show:
            p.show()
    return p


def plot_control_trajectory_curve(state_dataframe, static=True, keyword1="", show=True):
    """Plot control trajectory curves for different demand trajectories"""
    if static:
        p = sns.lineplot(x="time", y="control", hue="demand_trajectory", style=keyword1,
                         data=state_dataframe, legend=True)
        p.set_title("Evolution of installed capacities")
        p.set_xlabel("Time", fontsize=10)
        p.set_ylabel("Capacity installed (GW)", fontsize=10)
        if show:
            plt.show()
    else:
        p = px.line(state_dataframe, x="time", y="control", color="demand_trajectory", line_dash=keyword1,
                    title="Evolution of installed capacities", labels={
                "time": "Time",
                "state": "Capacity installed (GW)"
            }, )
        if show:
            p.show()
    return p


def plot_control_trajectory_curve_2param(state_dataframe, static=True, keyword1="", keyword2="", show=True):
    """Plot control trajectory curves for different demand trajectories"""
    # TODO: il faut ajouter keyword2 dans la version statique
    if static:
        p = sns.lineplot(x="time", y="control", hue="demand_trajectory", style=keyword1,
                         data=state_dataframe, legend=True)
        p.set_title("Evolution of installed capacities")
        p.set_xlabel("Time", fontsize=10)
        p.set_ylabel("Capacity installed (GW)", fontsize=10)
        if show:
            plt.show()
    else:
        p = px.line(state_dataframe, x="time", y="control", color="demand_trajectory", line_dash=keyword1,
                    facet_col=keyword2,
                    title="Evolution of installed capacities", labels={
                "time": "Time",
                "state": "Capacity installed (GW)"
            }, )
        if show:
            p.show()
    return p


def all_info_from_demand_trajectory(demand_trajectory, state_distribution, demand_states, x_cutoff, y_cutoff,
                                    Q_offshore):
    """Compiles state, cutoff points and Q offshore from a given demand trajectory. ATTENTION, here the demand
     trajectory includes demand from time -1 to t-1, which is necessary to determine state at time t, AND the final
     element in demand_trajectory corresponds to the value of demand at time t."""
    demand_time_steps = len(
        demand_trajectory)  # this includes demand at time -1, and final demand at time final_time
    final_time = demand_time_steps - 2  # from time t=0 to time t=final_time (we do not consider demand at time -1 here)
    # state_time = demand_time_steps - 2  # the last demand step does not define the state that we observe

    state_sun = state_distribution[final_time][0]
    state_wind = state_distribution[final_time][1]
    for i in range(final_time + 1):  # it starts with the value of demand at time -1
        state_sun = state_sun[demand_trajectory[i]]
        state_wind = state_wind[demand_trajectory[i]]
    final_state = demand_trajectory[
        final_time + 1]  # we take the last value of demand_trajectory corresponding to demand at time final_time
    D_average_final_t = demand_states[final_time][
        final_state]  # there is a gap of 1 when calculating D_average_final_t because demand at time -1 is not
    # included in demand_states
    x_cutoff_final_t = x_cutoff[final_time][final_state]  # because x_cutoff is not defined for time t=-1
    y_cutoff_final_t = y_cutoff[final_time][final_state]  # because y_cutoff is not defined for time t=-1
    Q_offshore_final_t = Q_offshore[final_time][final_state]  # because Q_offshore is not defined for time t=-1
    return state_sun, state_wind, D_average_final_t, x_cutoff_final_t, y_cutoff_final_t, Q_offshore_final_t


def create_hourly_dataframe(D_average, Q_sun, Q_wind, Q_offshore, x_cutoff, y_cutoff, weather_params, add_params):
    demand_centered, gamma_sun, gamma_wind_onshore, gamma_wind_offshore, gamma_river, hydro_prod, proba = (
        weather_params["demand"], weather_params["sun"], weather_params["onshore"],
        weather_params["offshore"], weather_params["river"], weather_params["hydro"], weather_params["proba"])
    market_price, voll, Q_pv_init, Q_onshore_init, Q_river = add_params['market_price'], add_params['voll'], \
                                                             add_params['Q_pv_init'], add_params[
                                                                 'Q_onshore_init'], add_params["Q_river"]
    price = np.minimum(supply_function_np_piecewise(np.maximum(
        D_average + demand_centered - (Q_sun + Q_pv_init) * gamma_sun - (Q_wind + Q_onshore_init) * gamma_wind_onshore
        - Q_offshore * gamma_wind_offshore - Q_river * gamma_river - hydro_prod, 0
    ), np.array(x_cutoff), np.array(y_cutoff), voll
    ),
        market_price)  # shape (n,) + (d,)*t where the t d corresponds to paths for variable Q
    weather_price_dataframe = pd.DataFrame({'demand_centered': demand_centered, 'price': price, 'gamma_sun': gamma_sun,
                                            'gamma_onshore': gamma_wind_onshore, 'gamma_offshore': gamma_wind_offshore,
                                            'river': gamma_river, 'hydro': hydro_prod, 'proba': proba
                                            })
    weather_price_dataframe["D_average"] = D_average
    weather_price_dataframe["demand"] = weather_price_dataframe["demand_centered"] + D_average
    weather_price_dataframe["residual_demand"] = (
            weather_price_dataframe["demand"]
            - (Q_sun + Q_pv_init) * weather_price_dataframe["gamma_sun"]
            - (Q_wind + Q_onshore_init) * weather_price_dataframe["gamma_onshore"]
            - Q_offshore * weather_price_dataframe["gamma_offshore"]
            - Q_river * weather_price_dataframe["river"]
            - weather_price_dataframe["hydro"]
    )
    return weather_price_dataframe


def create_hourly_dataframe_param(demand_trajectory, state_distribution_dict, demand_states, x_cutoff, y_cutoff,
                                  Q_offshore, weather_params, add_params, keyword, list_param):
    """Same as create_hourly_dataframe but for analyzing different settings based on one parameter"""
    weather_price_dataframe_tot = pd.DataFrame(
        {'demand_centered': pd.Series(dtype='float'), 'price': pd.Series(dtype='float'),
         'gamma_sun': pd.Series(dtype='float'),
         'gamma_onshore': pd.Series(dtype='float'), 'gamma_offshore': pd.Series(dtype='float'),
         'river': pd.Series(dtype='float'),
         'hydro': pd.Series(dtype='float'), 'proba': pd.Series(dtype='float'), 'D_average': pd.Series(dtype='float'),
         'demand': pd.Series(dtype='float'), 'residual_demand': pd.Series(dtype='float'),
         keyword: pd.Series(dtype='str')})
    for param in list_param:
        if keyword == "availnuc":  # in this case the cutoff points are modified so we need to calculate them again
            availnuc = float(keyword.replace("p", "."))
            joint_law, x_cutoff, y_cutoff, Q_offshore = data_process_with_scenarios(period="large",
                                                                                    avail_nuclear=availnuc)
        Q_sun, Q_wind, D_average_t, x_cutoff_t, y_cutoff_t, Q_offshore_t = all_info_from_demand_trajectory(
            demand_trajectory, state_distribution_dict[param], demand_states, x_cutoff, y_cutoff, Q_offshore)
        # print(f"Param: {param}, Q_sun: {Q_sun}, Q_wind: {Q_wind} ")
        weather_price_dataframe = create_hourly_dataframe(D_average_t, Q_sun, Q_wind, Q_offshore_t, x_cutoff_t,
                                                          y_cutoff_t, weather_params, add_params)
        weather_price_dataframe[keyword] = pd.Series([param for x in range(len(weather_price_dataframe.index))],
                                                     dtype=str)
        weather_price_dataframe_tot = pd.concat([weather_price_dataframe_tot, weather_price_dataframe],
                                                ignore_index=True)
    return weather_price_dataframe_tot


def create_hourly_dataframe_2param(demand_trajectory, state_distribution_dict, demand_states, x_cutoff, y_cutoff,
                                   Q_offshore, weather_params, add_params, keyword1, keyword2, list_param):
    """Same as create_hourly_dataframe but for analyzing different settings based on two parameters"""
    weather_price_dataframe_tot = pd.DataFrame(
        {'demand_centered': pd.Series(dtype='float'), 'price': pd.Series(dtype='float'),
         'gamma_sun': pd.Series(dtype='float'),
         'gamma_onshore': pd.Series(dtype='float'), 'gamma_offshore': pd.Series(dtype='float'),
         'river': pd.Series(dtype='float'),
         'hydro': pd.Series(dtype='float'), 'proba': pd.Series(dtype='float'), 'D_average': pd.Series(dtype='float'),
         'demand': pd.Series(dtype='float'), 'residual_demand': pd.Series(dtype='float'),
         keyword1: pd.Series(dtype='str'), keyword2: pd.Series(dtype='str')})
    for param in list_param:
        (param1, param2) = param
        # print(f"Param1 : {param1}, Param2: {param2}")
        if keyword1 == "availnuc":  # in this case the cutoff points are modified so we need to calculate them again
            availnuc = float(param1.replace("p", "."))
            joint_law, x_cutoff, y_cutoff, Q_offshore = data_process_with_scenarios(period="large",
                                                                                    avail_nuclear=availnuc)
        if keyword2 == "availnuc":
            availnuc = float(param2.replace("p", "."))
            joint_law, x_cutoff, y_cutoff, Q_offshore = data_process_with_scenarios(period="large",
                                                                                    avail_nuclear=availnuc)
        Q_sun, Q_wind, D_average_t, x_cutoff_t, y_cutoff_t, Q_offshore_t = all_info_from_demand_trajectory(
            demand_trajectory, state_distribution_dict[param], demand_states, x_cutoff, y_cutoff, Q_offshore)
        # print(f"Param: {param}, Q_sun: {Q_sun}, Q_wind: {Q_wind}, D_average_t: {D_average_t}")
        weather_price_dataframe = create_hourly_dataframe(D_average_t, Q_sun, Q_wind, Q_offshore_t, x_cutoff_t,
                                                          y_cutoff_t, weather_params, add_params)
        weather_price_dataframe[keyword1] = pd.Series([param1 for x in range(len(weather_price_dataframe.index))],
                                                      dtype=str)
        weather_price_dataframe[keyword2] = pd.Series([param2 for x in range(len(weather_price_dataframe.index))],
                                                      dtype=str)
        weather_price_dataframe_tot = pd.concat([weather_price_dataframe_tot, weather_price_dataframe],
                                                ignore_index=True)
    return weather_price_dataframe_tot


def get_load_duration_curve(weather_price_dataframe):
    """Process cumulative probability corresponding to demand ranked decreasingly"""
    weather_price_dataframe = weather_price_dataframe.sort_values(by="demand", ascending=False)
    weather_price_dataframe["percentage"] = weather_price_dataframe['proba'].cumsum()
    return weather_price_dataframe


def get_residual_load_duration_curve(weather_price_dataframe):
    """Process cumulative probability corresponding to demand ranked decreasingly"""
    weather_price_dataframe = weather_price_dataframe.sort_values(by="residual_demand", ascending=False)
    weather_price_dataframe["percentage"] = weather_price_dataframe['proba'].cumsum()
    return weather_price_dataframe


def get_price_duration_curve(weather_price_dataframe, multiple=False, keyword=""):
    """Process cumulative probability corresponding to price ranked decreasingly"""
    if multiple and keyword == "":
        print("When variable multiple is True, keyword should be specified to know how to perform groupby")
    if multiple:
        weather_price_dataframe["percentage"] = \
            weather_price_dataframe.sort_values(by="price", ascending=False).groupby(keyword)['proba'].cumsum()
        weather_price_dataframe = weather_price_dataframe.sort_values(by="price", ascending=False)
    else:
        weather_price_dataframe = weather_price_dataframe.sort_values(by="price", ascending=False)
        weather_price_dataframe["percentage"] = weather_price_dataframe['proba'].cumsum()
    return weather_price_dataframe


def get_price_duration_curve_2param(weather_price_dataframe, keyword1, keyword2):
    """Process cumulative probability corresponding to price ranked decreasingly"""
    weather_price_dataframe["percentage"] = \
        weather_price_dataframe.sort_values(by="price", ascending=False).groupby([keyword1, keyword2])['proba'].cumsum()
    weather_price_dataframe = weather_price_dataframe.sort_values(by="price", ascending=False)
    return weather_price_dataframe


def plot_residual_load_duration_curve(weather_price_dataframe, static=True, multiple=False, keyword=""):
    """Plot residual load duration curve"""
    if multiple and keyword == "":
        print("When variable multiple is True, keyword should be specified to know how to perform groupby")
    if multiple:
        if static:
            p = sns.lineplot(x="percentage", y="residual_demand", hue=keyword, data=weather_price_dataframe)
            p.set_title("Residual load-Duration Curve", fontsize=20)
            p.set_xlabel("Percentage (%)", fontsize=10)
            p.set_ylabel("Residual load (MWh)", fontsize=10)
            plt.show()
            # return p
        else:
            p = px.line(weather_price_dataframe, x="percentage", y="residual_demand", color=keyword,
                        title="Residual load (MWh)",
                        labels={
                            "percentage": "Percentage (%)",
                            "residual_demand": "Residual load (MWh)"
                        }, )
            p.show()
    else:
        if static:
            p = sns.lineplot(x="percentage", y="residual_demand", data=weather_price_dataframe)
            p.set_title("Residual load-Duration Curve", fontsize=20)
            p.set_xlabel("Percentage (%)", fontsize=10)
            p.set_ylabel("Residual load (MWh)", fontsize=10)
            plt.show()
            # return p
        else:
            p = px.line(weather_price_dataframe, x="percentage", y="residual_demand", title="Residual load (MWh)",
                        labels={
                            "percentage": "Percentage (%)",
                            "residual_demand": "Residual load (MWh)"
                        }, )
            p.show()
    return p


def plot_price_duration_curve(weather_price_dataframe, market_price_cap, cap=True, static=True, multiple=False,
                              keyword="", show=True):
    """Plot price duration curve"""
    if multiple and keyword == "":
        print("When variable multiple is True, keyword should be specified to know how to perform groupby")
    if multiple:
        if static:
            if cap:
                p = sns.lineplot(x="percentage", y="price", hue=keyword,
                                 data=weather_price_dataframe.loc[weather_price_dataframe.price < market_price_cap])
            else:
                p = sns.lineplot(x="percentage", y="price", hue=keyword,
                                 data=weather_price_dataframe)
            p.set_title("Price-Duration Curve", fontsize=20)
            p.set_xlabel("Percentage (%)", fontsize=10)
            p.set_ylabel("Price (EUR/MWh)", fontsize=10)
            if show:
                plt.show()
        else:
            if cap:
                p = px.line(weather_price_dataframe.loc[weather_price_dataframe.price < market_price_cap],
                            x="percentage",
                            y="price", color=keyword, title="Price-Duration Curve", labels={
                        "percentage": "Percentage (%)",
                        "residual_demand": "Price (EUR/MWh)"
                    }, )
            else:
                p = px.line(weather_price_dataframe, x="percentage", y="price", color=keyword,
                            title="Price-Duration Curve", labels={
                        "percentage": "Percentage (%)",
                        "residual_demand": "Price (EUR/MWh)"
                    }, )
            if show:
                p.show()
    else:
        if static:
            if cap:
                p = sns.lineplot(x="percentage", y="price",
                                 data=weather_price_dataframe.loc[weather_price_dataframe.price < market_price_cap])
            else:
                p = sns.lineplot(x="percentage", y="price",
                                 data=weather_price_dataframe)
            p.set_title("Price-Duration Curve", fontsize=20)
            p.set_xlabel("Percentage (%)", fontsize=10)
            p.set_ylabel("Price (EUR/MWh)", fontsize=10)
            if show:
                plt.show()
        else:
            if cap:
                p = px.line(weather_price_dataframe.loc[weather_price_dataframe.price < market_price_cap],
                            x="percentage",
                            y="price", title="Price-Duration Curve", labels={
                        "percentage": "Percentage (%)",
                        "residual_demand": "Price (EUR/MWh)"
                    }, )
            else:
                p = px.line(weather_price_dataframe, x="percentage", y="price", title="Price-Duration Curve", labels={
                    "percentage": "Percentage (%)",
                    "residual_demand": "Price (EUR/MWh)"
                }, )
            if show:
                p.show()
    return p


def plot_price_duration_curve_2param(weather_price_dataframe, market_price_cap, cap=True, static=True,
                                     keyword1="", keyword2="", show=True):
    """Plot price duration curve"""
    if static:
        if cap:
            p = sns.lineplot(x="percentage", y="price", hue=keyword1, style=keyword2,
                             data=weather_price_dataframe.loc[weather_price_dataframe.price < market_price_cap])
        else:
            p = sns.lineplot(x="percentage", y="price", hue=keyword1, style=keyword2,
                             data=weather_price_dataframe)
        p.set_title("Price-Duration Curve", fontsize=20)
        p.set_xlabel("Percentage (%)", fontsize=10)
        p.set_ylabel("Price (EUR/MWh)", fontsize=10)
        if show:
            plt.show()
    else:
        if cap:
            p = px.line(weather_price_dataframe.loc[weather_price_dataframe.price < market_price_cap],
                        x="percentage",
                        y="price", color=keyword1, line_dash=keyword2, title="Price-Duration Curve", labels={
                    "percentage": "Percentage (%)",
                    "residual_demand": "Price (EUR/MWh)"
                }, )
        else:
            p = px.line(weather_price_dataframe, x="percentage", y="price", color=keyword1, line_dash=keyword2,
                        title="Price-Duration Curve", labels={
                    "percentage": "Percentage (%)",
                    "residual_demand": "Price (EUR/MWh)"
                }, )
        if show:
            p.show()

    return p


def find_dispatchable_tec(price, y_cutoff):
    """Finds nearest smallest value to price in y_cutoff. Returns the index of the corresponding value in y_cutoff. """
    if price == 0:
        return 0  # price fixed by renewable
    else:
        return ((price - y_cutoff) <= 0).argmax() - 1


def name_dispatchable_tec(tec):
    """Would require something slightly more refined to include different types of gas."""
    if tec == 0:
        return "REN"
    elif tec == 1:
        return "nuclear"
    else:
        return "gas"


def dispatchable_tec(weather_price_dataframe, x_cutoff, y_cutoff, Q_sun, Q_onshore, Q_offshore, Q_river):
    weather_price_dataframe["tec_dispatchable"] = weather_price_dataframe.apply(
        lambda row: find_dispatchable_tec(row["price"], y_cutoff), axis=0)
    weather_price_dataframe["tec_dispatchable"] = weather_price_dataframe.apply(
        lambda row: name_dispatchable_tec(row["tec_dispatchable"]), axis=0)
    weather_price_dataframe["generation_sun"] = weather_price_dataframe["gamma_sun"] * Q_sun
    weather_price_dataframe["generation_onshore"] = weather_price_dataframe["gamma_onshore"] * Q_onshore
    weather_price_dataframe["generation_offshore"] = weather_price_dataframe["gamma_offshore"] * Q_offshore
    weather_price_dataframe["generation_river"] = weather_price_dataframe["river"] * Q_river
    # TODO: il faut ajouter une fonction pour indiquer la capacité appelée en dispatchable
    return weather_price_dataframe


def plot_average_mix(weather_price_dataframe):
    """Function to plot the average generation mix"""
    # TODO: fonction a faire
    # weather_price_dataframe["generation"] = weather_price_dataframe[]
    return 0

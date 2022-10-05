from investment.optimization_model import *
from investment.data_loading import data_process_with_scenarios
from investment.demand_model import demand_model
import seaborn as sns
import datetime

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    T = 5
    Tprime = 8

    premium = np.full((T,), 0)
    additional_parameters = {'market_price': 10000,
                             'voll': 15000,
                             'discount_rate': 0.04,
                             'nu_deval': 0.15,
                             'Delta_t': 5 * 365 * 24,
                             'c_tilde_sun': 60000,
                             'c_tilde_wind': 120000,
                             'Q_pv_init': 10,
                             'Q_onshore_init': 18,
                             'Q_river': 10,
                             'investment_costs_sun': [6 * 1e5, 5.5 * 1e5, 5.2 * 1e5, 5 * 1e5, 5 * 1e5],
                             'investment_costs_wind': [12.5 * 1e5, 12 * 1e5, 11.1 * 1e5, 10.3 * 1e5, 9.8 * 1e5]}

    # General information
    trans_matrix, demand_states = demand_model()
    nb_scenario_gas = 100
    variation_gas = np.random.uniform(-20, 20, nb_scenario_gas)
    joint_law, x_cutoff, y_cutoff, Q_offshore = data_process_with_scenarios(period="large", avail_nuclear=0.9)
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
    demand_centered_tot, pv_tot, onshore_tot, offshore_tot, river_tot, hydro_prod_tot, probability_tot, year_tot = load_all_weather_data(
        1980, 2019)
    weather_tot_params = {
        "demand": demand_centered_tot,
        "sun": pv_tot,
        "onshore": onshore_tot,
        "offshore": offshore_tot,
        "river": river_tot,
        "hydro": hydro_prod_tot,
        "proba": probability_tot,
        "years": year_tot
    }

    strategy_others = np.array([[[10, 20, 30],
                                 [10, 20, 30],
                                 [10, 20, 30]],
                                [[10, 20, 30],
                                 [10, 20, 30],
                                 [10, 20, 30]]])
    q_next = np.array([[[5, 10, 15],
                        [5, 10, 15],
                        [5, 10, 15]],
                       [[5, 10, 15],
                        [5, 10, 15],
                        [5, 10, 15], ],
                       [[5, 10, 15],
                        [5, 10, 15],
                        [5, 10, 15]]])  # at time t=2
    t = 1
    # start = time.time()
    # opt_strategy_t1_b0p5 = recursive_optimal_control(t, q_next, trans_matrix, demand_states, beta=0.5, alpha=0.05,
    #                                                  start=1980, end=2019, gas_scenarios=variation_gas,
    #                                                  Q_t=strategy_others,
    #                                                  premium=0, weather_params=weather_params,
    #                                                  weather_tot_params=weather_tot_params,
    #                                                  Q_offshore_t=Q_offshore[t], tec="sun", x_cutoff_t=x_cutoff[1],
    #                                                  y_cutoff_t=y_cutoff[1], add_params=additional_parameters)
    # opt_strategy_t1_b0p9 = recursive_optimal_control(t, q_next, trans_matrix, demand_states, beta=0.9, alpha=0.05,
    #                                                  start=1980, end=2019, gas_scenarios=variation_gas,
    #                                                  Q_t=strategy_others,
    #                                                  premium=0, weather_params=weather_params,
    #                                                  weather_tot_params=weather_tot_params,
    #                                                  Q_offshore_t=Q_offshore[t], tec="sun", x_cutoff_t=x_cutoff[1],
    #                                                  y_cutoff_t=y_cutoff[1], add_params=additional_parameters)
    # end = time.time()
    # print(round(end - start, 10))

    # Testing updating optimal control
    # start = time.time()
    # strategy_others = []
    # for t in range(0, T, 1):
    #     strategy_t = np.full((2,) + (3,) * (t + 1), 10 * (t + 1))
    #     strategy_others.append(strategy_t)
    # opt_control_b0p5 = find_optimal_control(T, Tprime, trans_matrix, demand_states, beta=0.5, alpha=0.05, start=1980,
    #                                    end=2019, gas_scenarios=variation_gas,
    #                                    Q=strategy_others, premium=premium, weather_params=weather_params,
    #                                    weather_tot_params=weather_tot_params, Q_offshore=Q_offshore, tec="sun",
    #                                    x_cutoff=x_cutoff, y_cutoff=y_cutoff, add_params=additional_parameters)
    #
    # opt_control_b0p9 = find_optimal_control(T, Tprime, trans_matrix, demand_states, beta=0.9, alpha=0.05, start=1980,
    #                                    end=2019, gas_scenarios=variation_gas,
    #                                    Q=strategy_others, premium=premium, weather_params=weather_params,
    #                                    weather_tot_params=weather_tot_params, Q_offshore=Q_offshore, tec="sun",
    #                                    x_cutoff=x_cutoff, y_cutoff=y_cutoff, add_params=additional_parameters)
    # end = time.time()
    # print(f"Time for execution: {round(end - start, 10)}")


    # Initialize strategy
    strategy_init = []
    # strategy_RTE_sun = [25, 40, 60, 80, 100]
    # strategy_RTE_wind = [25, 35, 40, 50, 55]
    strategy_sun_init = [0, 10, 20, 40, 60]  # we initialize with values obtained with no premium
    strategy_wind_init = [0, 15, 40, 60, 100]
    # strategy_RTE_sun = [0, 0, 0, 0, 0]
    # strategy_RTE_wind = [500, 600, 0, 0, 0]
    for t in range(0, T, 1):
        strategy_t_sun = np.full((3,) * (t + 1), strategy_sun_init[t])
        strategy_t_wind = np.full((3,) * (t + 1), strategy_wind_init[t])
        strategy_t = np.array([strategy_t_sun, strategy_t_wind])
        strategy_init.append(strategy_t)

    N = 10
    beta = 1
    # Run algorithm
    state_distribution, objective_gap, index_objective_gap = fictitious_play(N=N, T=T, Tprime=Tprime,
                                                                             state_init=strategy_init,
                                                                             trans_matrix=trans_matrix,
                                                                             demand_states=demand_states,
                                                                             beta=beta,
                                                                             alpha=0.05,
                                                                             start=1980,
                                                                             end=2019,
                                                                             gas_scenarios=variation_gas,
                                                                             premium=premium,
                                                                             weather_params=weather_params,
                                                                             weather_tot_params=weather_tot_params,
                                                                             Q_offshore=Q_offshore,
                                                                             x_cutoff=x_cutoff,
                                                                             y_cutoff=y_cutoff,
                                                                             add_params=additional_parameters,
                                                                             convergence="wolfe")

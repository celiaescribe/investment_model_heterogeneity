from investment.optimization_model import *
from investment.data_loading import data_process_with_scenarios, load_all_weather_data
from investment.demand_model import demand_model, demand_model_no_uncertainty
from investment.post_processing import process_output
import seaborn as sns
from datetime import datetime
from pathlib import Path
import json
import argparse
from scipy.stats import binom
from investment.utils import generate_distributions_small, generate_distribution_large, generate_distribution_beta_binomial, generate_distribution_beta_binomial_large, generate_distribution_beta_binomial_verylarge

parser = argparse.ArgumentParser(description='Run investment model.')
parser.add_argument("-d", "--discount", type=float, help="discount rate")
# parser.add_argument("-b", "--beta", type=float, help="risk aversion")
parser.add_argument("-n", "--nu", type=float, help="devaluation rate")
# parser.add_argument("-nuc", "--nuclear", type=float, help="availability of nuclear")
parser.add_argument("--cvar", type=float, help="CVAR parameter")
parser.add_argument("-p", "--premium", type=float, help="premium parameter")
parser.add_argument("--cap", type=float, help="Market price cap")
parser.add_argument("-i", "--iterations", type=int, help="Number of iterations")
parser.add_argument("-dir", "--directory", type=str, help="Name of created directory")
parser.add_argument("-m", "--message", type=str, help="message describing experiment")
args = parser.parse_args()

# discount_rate_yearly, beta, nu_deval, availnuc, cvar_level, premium_value, market_cap, N, directory_suffix, message = args.discount, args.beta, \
#                                                                                                                       args.nu, \
#                                                                                                                       args.nuclear, args.cvar, args.premium, args.cap, \
#                                                                                                                       args.iterations, args.directory, \
#                                                                                                                       args.message
discount_rate_yearly, nu_deval, cvar_level, premium_value, market_cap, N, directory_suffix, message = args.discount, args.nu, \
                                                                                                      args.cvar, args.premium, args.cap, \
                                                                                                      args.iterations, args.directory, \
                                                                                                      args.message

# discount_rate_yearly, beta, nu_deval, availnuc, cvar_level, N = 0.04, 1, 0.15, 1, 0.05, 20


list_beta = [0.5, 0.9]
list_avail_nuc = [0.9]

gamma, weights = generate_distribution_beta_binomial()

# First test: we only try one value for beta
# list_beta = [0.6]
# weights = {
#     'singleton': [1]
# }

for distrib in list(weights.keys()):
    list_weight_gamma = weights[distrib]
    list_gamma = gamma[distrib]
    for beta in list_beta:
        for availnuc in list_avail_nuc:
            print(
                f"Model with weights {distrib}, beta {beta}, discount {discount_rate_yearly}, nu {nu_deval}, nuclear {availnuc}, cvar {cvar_level}, premium {premium_value}, market cap {market_cap}",
                flush=True)
            # Create directory if does not exists
            day = datetime.now().strftime("%m%d")
            directory_name = "outputs/" + day + "_" + directory_suffix
            Path(directory_name).mkdir(parents=True, exist_ok=True)

            T = 5
            Tprime = 8
            time_step = 5  # number of years between investment decisions
            discount_rate = time_step * discount_rate_yearly
            # market_price = 10000  # market price cap
            Delta_t = time_step * 365 * 24  # number of hours in ergodic theorem
            voll = 15000  # value of loss load
            c_tilde_sun = 60000  # sun, wind
            c_tilde_wind = 120000
            investment_costs_sun = [6 * 1e5, 5.5 * 1e5, 5.2 * 1e5, 5 * 1e5, 5 * 1e5]
            investment_costs_wind = [12.5 * 1e5, 12 * 1e5, 11.1 * 1e5, 10.3 * 1e5, 9.8 * 1e5]
            Q_pv_init = 10
            Q_onshore_init = 18
            Q_river = 10

            # premium_value = 0
            premium = np.full((T,), premium_value)
            additional_parameters = {'market_price': market_cap,
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
            trans_matrix, demand_states = demand_model_no_uncertainty()  # TODO: ligne à modifier si besoin

            # Scenario gas prices
            np.random.seed(123)
            nb_scenario_gas = 100
            variation_gas = np.random.uniform(-20, 20, nb_scenario_gas)

            # Weather params
            joint_law, x_cutoff, y_cutoff, Q_offshore = data_process_with_scenarios(period="large", avail_nuclear=availnuc)
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
                1980, 2019, imports=False)  # we do not consider imports for now
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
            # Initialize strategy
            strategy_init = []
            # strategy_RTE_sun = [25, 40, 60, 80, 100]
            # strategy_RTE_wind = [25, 35, 40, 50, 55]
            strategy_sun_init = [0, 10, 20, 40, 60]  # we initialize with values obtained with no premium
            strategy_wind_init = [0, 15, 40, 60, 100]

            for t in range(0, T, 1):
                strategy_t_sun = np.full((3,) * (t + 1), strategy_sun_init[t])
                strategy_t_wind = np.full((3,) * (t + 1), strategy_wind_init[t])
                strategy_t = np.array([strategy_t_sun, strategy_t_wind])
                strategy_init.append(strategy_t)

            # Run algorithm
            state_distribution, objective_gap, index_objective_gap = fictitious_play(N=N, T=T,
                                                                                     Tprime=Tprime,
                                                                                     state_init=strategy_init,
                                                                                     trans_matrix=trans_matrix,
                                                                                     demand_states=demand_states,
                                                                                     beta=beta,
                                                                                     list_gamma=list_gamma,
                                                                                     list_weight_gamma=list_weight_gamma,
                                                                                     alpha=cvar_level,
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
                                                                                     convergence="wolfe",
                                                                                     deval=True)

            # Process output
            subdirectory_name = directory_name + f"/weight{distrib}_beta{beta}_discount{discount_rate_yearly}_nu{nu_deval}_premium{premium_value}_availnuc{availnuc}_cvar{cvar_level}_cap{market_cap}"
            subdirectory_name = subdirectory_name.replace(".", "p")  # replace all points in file by p to avoid problems
            subdirectory_name = subdirectory_name.replace("-",
                                                          "m")  # replace all minus in file by symbol m to avoid problems
            Path(subdirectory_name).mkdir(parents=True, exist_ok=True)

            process_output(T, state_distribution, objective_gap, index_objective_gap, list_gamma, list_weight_gamma,
                           subdirectory_name, additional_parameters)

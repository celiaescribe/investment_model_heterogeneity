import numpy as np


def param_trans_matrix(alpha1, alpha2, gamma1, gamma2):
    return [[alpha1, 1 - alpha1, 0],
            [gamma1, 1 - gamma1 - gamma2, gamma2],
            [0, 1 - alpha2, alpha2]]


def demand_model():
    # For now, we consider homogeneous transitions from one state to the other
    trans_matrix_2025 = [[0.25, 0.5, 0.25],
                         [0.25, 0.5, 0.25],
                         [0.25, 0.5, 0.25]]
    trans_matrix_2030 = param_trans_matrix(alpha1=0.75, alpha2=0.75, gamma1=0.25, gamma2=0.25)
    trans_matrix_2035 = trans_matrix_2030
    trans_matrix_2040 = trans_matrix_2030
    trans_matrix_2045 = trans_matrix_2030

    trans_matrix = np.array([trans_matrix_2025, trans_matrix_2030, trans_matrix_2035, trans_matrix_2040,
                             trans_matrix_2045])

    # New values
    states_2025 = [56.07, 58.1, 60.5]
    states_2030 = [57.94, 62, 66.8]
    states_2035 = [59.80, 65.9, 73.1]
    states_2040 = [61.67, 69.8, 79.4]
    states_2045 = [63.4, 73.7, 85.6]

    demand_states = np.array([states_2025,
                              states_2030,
                              states_2035,
                              states_2040,
                              states_2045])

    # trans_matrix = [trans_matrix_2025, trans_matrix_2030, trans_matrix_2035, trans_matrix_2040, trans_matrix_2045]
    return trans_matrix, demand_states


def demand_model_no_uncertainty():
    """Demand model where we only stay on average trajectory"""
    trans_matrix_2025 = [[0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0]]
    trans_matrix_2030 = param_trans_matrix(alpha1=0, alpha2=0, gamma1=0, gamma2=0)
    trans_matrix_2035 = trans_matrix_2030
    trans_matrix_2040 = trans_matrix_2030
    trans_matrix_2045 = trans_matrix_2030

    trans_matrix = np.array([trans_matrix_2025, trans_matrix_2030, trans_matrix_2035, trans_matrix_2040,
                             trans_matrix_2045])

    # New values
    states_2025 = [56.07, 58.1, 60.5]
    states_2030 = [57.94, 62, 66.8]
    states_2035 = [59.80, 65.9, 73.1]
    states_2040 = [61.67, 69.8, 79.4]
    states_2045 = [63.4, 73.7, 85.6]

    demand_states = np.array([states_2025,
                              states_2030,
                              states_2035,
                              states_2040,
                              states_2045])

    # trans_matrix = [trans_matrix_2025, trans_matrix_2030, trans_matrix_2035, trans_matrix_2040, trans_matrix_2045]
    return trans_matrix, demand_states


def demand_model_param(alpha1, alpha2, gamma1, gamma2):
    trans_matrix_2025 = [[gamma1, 1 - gamma1 - gamma2, gamma2],
                         [gamma1, 1 - gamma1 - gamma2, gamma2],
                         [gamma1, 1 - gamma1 - gamma2, gamma2]]
    trans_matrix_2030 = param_trans_matrix(alpha1=alpha1, alpha2=alpha2, gamma1=gamma1, gamma2=gamma2)
    trans_matrix_2035 = trans_matrix_2030
    trans_matrix_2040 = trans_matrix_2030
    trans_matrix_2045 = trans_matrix_2030

    trans_matrix = np.array([trans_matrix_2025, trans_matrix_2030, trans_matrix_2035, trans_matrix_2040,
                             trans_matrix_2045])

    states_2025 = [54.6, 56.6, 59]
    states_2030 = [56.2, 60.2, 65]
    states_2035 = [57.8, 63.8, 71]
    states_2040 = [59.4, 67.4, 77]
    states_2045 = [61, 71, 84]

    demand_states = np.array([states_2025,
                              states_2030,
                              states_2035,
                              states_2040,
                              states_2045])

    return trans_matrix, demand_states


def demand_model_param_evolving(alpha1, alpha2, gamma1, gamma2):
    trans_matrix_2025 = [[gamma1, 1 - gamma1 - gamma2, gamma2],
                         [gamma1, 1 - gamma1 - gamma2, gamma2],
                         [gamma1, 1 - gamma1 - gamma2, gamma2]]
    trans_matrix_2030 = param_trans_matrix(alpha1=alpha1, alpha2=alpha2, gamma1=gamma1, gamma2=gamma2)
    trans_matrix_2035 = trans_matrix_2030
    trans_matrix_2040 = trans_matrix_2030
    trans_matrix_2045 = trans_matrix_2030

    trans_matrix = np.array([trans_matrix_2025, trans_matrix_2030, trans_matrix_2035, trans_matrix_2040,
                             trans_matrix_2045])

    states_2025 = [54.6, 56.6, 59]
    states_2030 = [56.2, 60.2, 65]
    states_2035 = [57.8, 63.8, 71]
    states_2040 = [59.4, 67.4, 77]
    states_2045 = [61, 71, 84]

    demand_states = np.array([states_2025,
                              states_2030,
                              states_2035,
                              states_2040,
                              states_2045])

    return trans_matrix, demand_states

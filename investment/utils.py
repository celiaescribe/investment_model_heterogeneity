import numpy as np
from scipy.stats import betabinom


def generate_distributions_small():
    list_gamma_2dirac_uniform = [0.9, 1.1]
    list_gamma_2dirac_NonUniformHigh = [0.9, 1.1]
    list_gamma_2dirac_NonUniformLow = [0.9, 1.1]
    list_gamma_3dirac = [0.8, 1, 1.2]
    list_gamma = list(np.arange(0.9, 1.1, 0.02))

    list_weight_gamma_2dirac_uniform = [1 / 2, 1 / 2]
    list_weight_gamma_2dirac_NonUniformHigh = [1 / 4, 3 / 4]
    list_weight_gamma_2dirac_NonUniformLow = [3 / 4, 1 / 4]
    list_weight_gamma_3dirac_uniform = [1 / 3, 1 / 3, 1 / 3]

    dict_weight_gamma_singleton = {}
    dict_gamma_singleton = {}
    for i in range(len(list_gamma)):  # creating values for dirac
        gamma = list_gamma[i]
        name_weight = f"singleton{round(gamma, 2)}"
        name_weight = name_weight.replace(".", "p")
        dict_weight_gamma_singleton[name_weight] = [1]
        dict_gamma_singleton[name_weight] = [gamma]

    gamma = {
        "2diracequal": list_gamma_2dirac_uniform,
        "2DiracNonUniformHigh": list_gamma_2dirac_NonUniformHigh,
        "2DiracNonUniformLow": list_gamma_2dirac_NonUniformLow,
        "3dirac": list_gamma_3dirac
    }
    weights = {
        "2diracequal": list_weight_gamma_2dirac_uniform,
        "2DiracNonUniformHigh": list_weight_gamma_2dirac_NonUniformHigh,
        "2DiracNonUniformLow": list_weight_gamma_2dirac_NonUniformLow,
        '3dirac': list_weight_gamma_3dirac_uniform
    }

    gamma.update(dict_gamma_singleton)
    weights.update(dict_weight_gamma_singleton)

    return gamma, weights


def generate_distribution_large():
    list_gamma_2dirac_uniform = [0.7, 1.3]
    list_gamma_2dirac_NonUniformHigh = [0.7, 1.3]
    list_gamma_2dirac_NonUniformLow = [0.7, 1.3]
    # list_gamma_3dirac = [0.8, 1, 1.2]
    list_gamma = list(np.arange(0.7, 1.3, 0.05))

    list_weight_gamma_2dirac_uniform = [1 / 2, 1 / 2]
    list_weight_gamma_2dirac_NonUniformHigh = [1 / 4, 3 / 4]
    list_weight_gamma_2dirac_NonUniformLow = [3 / 4, 1 / 4]
    # list_weight_gamma_3dirac_uniform = [1 / 3, 1 / 3, 1 / 3]

    dict_weight_gamma_singleton = {}
    dict_gamma_singleton = {}
    for i in range(len(list_gamma)):  # creating values for dirac
        gamma = list_gamma[i]
        name_weight = f"singleton{round(gamma, 2)}"
        name_weight = name_weight.replace(".", "p")
        dict_weight_gamma_singleton[name_weight] = [1]
        dict_gamma_singleton[name_weight] = [gamma]

    gamma = {
        "2diracequal": list_gamma_2dirac_uniform,
        "2DiracNonUniformHigh": list_gamma_2dirac_NonUniformHigh,
        "2DiracNonUniformLow": list_gamma_2dirac_NonUniformLow
    }
    weights = {
        "2diracequal": list_weight_gamma_2dirac_uniform,
        "2DiracNonUniformHigh": list_weight_gamma_2dirac_NonUniformHigh,
        "2DiracNonUniformLow": list_weight_gamma_2dirac_NonUniformLow
    }

    gamma.update(dict_gamma_singleton)
    weights.update(dict_weight_gamma_singleton)
    return gamma, weights


def generate_distribution_beta_binomial(n, a, b):
    r_values = list(range(n + 1))
    pmf_values = [betabinom.pmf(r, n, a, b) for r in r_values]
    return pmf_values
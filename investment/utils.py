import numpy as np
from scipy.stats import betabinom

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from pickle import load

sns.set_theme(context="talk", style="white")


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


def generate_weights_beta_binomial(n, a, b):
    r_values = list(range(n + 1))
    pmf_values = [betabinom.pmf(r, n, a, b) for r in r_values]
    return pmf_values


def generate_distribution_beta_binomial():
    """Generates distribution including beta binomial distribution"""
    list_gamma_2dirac_uniform = [0.9, 1.1]
    list_gamma_2dirac_NonUniformHigh = [0.9, 1.1]
    list_gamma_2dirac_NonUniformLow = [0.9, 1.1]
    list_gamma = list(np.arange(0.9, 1.1, 0.02))
    n = len(list_gamma) - 1  # used for beta binomial distribution

    list_weight_gamma_2dirac_uniform = [1 / 2, 1 / 2]
    list_weight_gamma_2dirac_NonUniformHigh = [1 / 4, 3 / 4]
    list_weight_gamma_2dirac_NonUniformLow = [3 / 4, 1 / 4]
    weigts_beta_binomial_symmetric = generate_weights_beta_binomial(n, a=0.5, b=0.5)
    weigts_beta_binomial_low = generate_weights_beta_binomial(n, a=0.15, b=0.3)
    weigts_beta_binomial_high = generate_weights_beta_binomial(n, a=0.3, b=0.15)

    gamma = {
        "2diracequal": list_gamma_2dirac_uniform,
        "2DiracNonUniformHigh": list_gamma_2dirac_NonUniformHigh,
        "2DiracNonUniformLow": list_gamma_2dirac_NonUniformLow,
        "BetaBinomialSymmetric": list_gamma,
        "BetaBinomialLow": list_gamma,
        "BetaBinomialHigh": list_gamma,
    }
    weights = {
        "2diracequal": list_weight_gamma_2dirac_uniform,
        "2DiracNonUniformHigh": list_weight_gamma_2dirac_NonUniformHigh,
        "2DiracNonUniformLow": list_weight_gamma_2dirac_NonUniformLow,
        "BetaBinomialSymmetric": weigts_beta_binomial_symmetric,
        "BetaBinomialLow": weigts_beta_binomial_low,
        "BetaBinomialHigh": weigts_beta_binomial_high,
    }

    dict_weight_gamma_singleton = {}
    dict_gamma_singleton = {}
    for i in range(len(list_gamma)):  # creating values for dirac
        name_weight = f"singleton{round(list_gamma[i], 2)}"
        name_weight = name_weight.replace(".", "p")
        dict_weight_gamma_singleton[name_weight] = [1]
        dict_gamma_singleton[name_weight] = [list_gamma[i]]

    gamma.update(dict_gamma_singleton)
    weights.update(dict_weight_gamma_singleton)

    return gamma, weights


def generate_distribution_beta_binomial_large():
    """Generates distribution including beta binomial distribution"""
    list_gamma_2dirac_uniform = [0.7, 1.3]
    list_gamma_2dirac_NonUniformHigh = [0.7, 1.3]
    list_gamma_2dirac_NonUniformLow = [0.7, 1.3]
    list_gamma = list(np.arange(0.7, 1.3, 0.05))
    n = len(list_gamma) - 1  # used for beta binomial distribution

    list_weight_gamma_2dirac_uniform = [1 / 2, 1 / 2]
    list_weight_gamma_2dirac_NonUniformHigh = [1 / 4, 3 / 4]
    list_weight_gamma_2dirac_NonUniformLow = [3 / 4, 1 / 4]
    weigts_beta_binomial_symmetric = generate_weights_beta_binomial(n, a=0.5, b=0.5)
    weigts_beta_binomial_low = generate_weights_beta_binomial(n, a=0.15, b=0.3)
    weigts_beta_binomial_high = generate_weights_beta_binomial(n, a=0.3, b=0.15)

    gamma = {
        "2diracequal": list_gamma_2dirac_uniform,
        "2DiracNonUniformHigh": list_gamma_2dirac_NonUniformHigh,
        "2DiracNonUniformLow": list_gamma_2dirac_NonUniformLow,
        "BetaBinomialSymmetric": list_gamma,
        "BetaBinomialLow": list_gamma,
        "BetaBinomialHigh": list_gamma,
    }
    weights = {
        "2diracequal": list_weight_gamma_2dirac_uniform,
        "2DiracNonUniformHigh": list_weight_gamma_2dirac_NonUniformHigh,
        "2DiracNonUniformLow": list_weight_gamma_2dirac_NonUniformLow,
        "BetaBinomialSymmetric": weigts_beta_binomial_symmetric,
        "BetaBinomialLow": weigts_beta_binomial_low,
        "BetaBinomialHigh": weigts_beta_binomial_high,
    }

    dict_weight_gamma_singleton = {}
    dict_gamma_singleton = {}
    for i in range(len(list_gamma)):  # creating values for dirac
        name_weight = f"singleton{round(list_gamma[i], 2)}"
        name_weight = name_weight.replace(".", "p")
        dict_weight_gamma_singleton[name_weight] = [1]
        dict_gamma_singleton[name_weight] = [list_gamma[i]]

    gamma.update(dict_gamma_singleton)
    weights.update(dict_weight_gamma_singleton)

    return gamma, weights


def generate_distribution_beta_binomial_verylarge():
    """Generates distribution including beta binomial distribution"""
    list_gamma_2dirac_uniform = [0.5, 1.5]
    list_gamma_2dirac_NonUniformHigh = [0.5, 1.5]
    list_gamma_2dirac_NonUniformLow = [0.5, 1.5]
    list_gamma = list(np.arange(0.5, 1.75, 0.25))
    # list_gamma = list(np.arange(0.5, 1.55, 0.05))
    n = len(list_gamma) - 1  # used for beta binomial distribution

    list_gamma_singleton = list(np.arange(0.5, 1.55, 0.05))

    list_weight_gamma_2dirac_uniform = [1 / 2, 1 / 2]
    list_weight_gamma_2dirac_NonUniformHigh = [1 / 4, 3 / 4]
    list_weight_gamma_2dirac_NonUniformLow = [3 / 4, 1 / 4]
    weigts_beta_binomial_symmetric = generate_weights_beta_binomial(n, a=0.5, b=0.5)
    weigts_beta_binomial_low = generate_weights_beta_binomial(n, a=0.15, b=0.3)
    weigts_beta_binomial_high = generate_weights_beta_binomial(n, a=0.3, b=0.15)

    gamma = {
        "2DiracDqual": list_gamma_2dirac_uniform,
        "2DiracNonUniformHigh": list_gamma_2dirac_NonUniformHigh,
        "2DiracNonUniformLow": list_gamma_2dirac_NonUniformLow,
        "BetaBinomialSymmetric": list_gamma,
        "BetaBinomialLow": list_gamma,
        "BetaBinomialHigh": list_gamma,
    }
    weights = {
        "2DiracEqual": list_weight_gamma_2dirac_uniform,
        "2DiracNonUniformHigh": list_weight_gamma_2dirac_NonUniformHigh,
        "2DiracNonUniformLow": list_weight_gamma_2dirac_NonUniformLow,
        "BetaBinomialSymmetric": weigts_beta_binomial_symmetric,
        "BetaBinomialLow": weigts_beta_binomial_low,
        "BetaBinomialHigh": weigts_beta_binomial_high,
    }

    dict_weight_gamma_singleton = {}
    dict_gamma_singleton = {}
    for i in range(len(list_gamma_singleton)):  # creating values for dirac
        name_weight = f"singleton{round(list_gamma_singleton[i], 2)}"
        name_weight = name_weight.replace(".", "p")
        dict_weight_gamma_singleton[name_weight] = [1]
        dict_gamma_singleton[name_weight] = [list_gamma_singleton[i]]

    gamma.update(dict_gamma_singleton)
    weights.update(dict_weight_gamma_singleton)

    return gamma, weights


def make_line_plot(df, subset=None, y_label=None, colors=None, format_y=lambda y, _: y, save=None, rotation=None,
                   x_ticks=None, index_int=True, dict_legend=None, legend=True, figsize=None):
    if save is None:
        if figsize is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:  # we change figure size when saving figure
        if figsize is None:
            fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
    if index_int:
        df.index = df.index.astype(int)
    if subset is None:
        if colors is None:
            df.plot.line(ax=ax)
        else:
            df.plot.line(ax=ax, color=colors)
    else:
        if colors is None:
            df[subset].plot.line(ax=ax)
        else:
            df[subset].plot.line(ax=ax, color=colors)

    if x_ticks is None:
        ax = format_ax(ax, title=y_label, x_ticks=df.index, format_y=format_y, rotation=rotation)
    else:
        ax = format_ax(ax, title=y_label, x_ticks=x_ticks, format_y=format_y, rotation=rotation)

    if legend:
        format_legend(ax, dict_legend=dict_legend)
    else:
        ax.get_legend().remove()

    save_fig(fig, save=save)


def make_line_plots(df, key_groupby, y_label, subset_groupby=None, format_y=lambda y, _: y, colors=None, x_ticks=None, index_int=True, save=None, rotation=None,
                    figsize=None):
    """Make line plot by combining different scenarios."""
    if save is None:
        if figsize is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:  # we change figure size when saving figure
        if figsize is None:
            fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
    if subset_groupby is not None:
        df = df.loc[df[key_groupby].isin(subset_groupby)]
    for key, g in df.groupby(key_groupby):
        g = g.drop(columns=[key_groupby]).squeeze().rename(key)
        if index_int:
            g.index = g.index.astype(int)
        if colors is None:
            g.plot.line(ax=ax)
        else:
            g.plot.line(ax=ax, color=colors)

    if x_ticks is None:
        ax = format_ax(ax, title=y_label, x_ticks=g.index, format_y=format_y, rotation=rotation)
    else:
        ax = format_ax(ax, title=y_label, x_ticks=x_ticks, format_y=format_y, rotation=rotation)

    format_legend(ax)

    save_fig(fig, save=save)


def format_ax(ax: plt.Axes, title=None, y_label=None, x_label=None, x_ticks=None, format_y=lambda y, _: y,
              rotation=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
    if x_ticks is not None:
        ax.set_xticks(ticks=x_ticks, labels=x_ticks)
    if rotation is not None:
        ax.set_xticklabels(ax.get_xticks(), rotation=rotation)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if title is not None:
        ax.set_title(title)

    return ax


def format_legend(ax, dict_legend=None):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    if dict_legend is not None:
        current_labels = ax.get_legend_handles_labels()[1]
        new_labels = [dict_legend[e] for e in current_labels]
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), labels=new_labels, frameon=False)
    else:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)


def save_fig(fig, save=None, bbox_inches='tight'):
    if save is not None:
        fig.savefig(save, bbox_inches=bbox_inches)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    gamma, weights = generate_distribution_beta_binomial_large()
    distrib = "BetaBinomialHigh"
    gamma_betabinomhigh = np.array(gamma[distrib])
    weight_betabinomhigh = np.array(weights[distrib])

    esperance = sum(gamma_betabinomhigh * weight_betabinomhigh)
    esperance_du_carre = sum(gamma_betabinomhigh**2 * weight_betabinomhigh)
    esperance_au_carre = sum(gamma_betabinomhigh * weight_betabinomhigh)**2

    print(f"Esperance: {esperance}")
    print(f"Esperance du carre: {esperance_du_carre}")
    print(f"Esperance au carre: {esperance_au_carre}")


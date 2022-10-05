import pandas as pd
import numpy as np
import datetime


def process_hourly_data(demand, pv, onshore, offshore, river, production, imports=False):
    """
    Processes all hourly data to provide a final global dataframe
    :param demand: Dataframe with hourly demand data which must includes columns ["hour", "demand_FR"]
    :param pv: Dataframe with hourly pv data which must includes columns ["hour", "FR"]
    :param onshore: Dataframe with hourly onshore data which must includes columns ["hour", "FR"]
    :param offshore: Dataframe with hourly offshore data which must includes columns ["hour", "FR"]
    :param river: Dataframe with hourly river data which must includes columns ["hour", "FR"]
    :param production: Dataframe with hourly data on lake, river, imports, exports...
    :return: pd.Dataframe
    """
    # Processing demand
    demand = demand[["hour", "FR"]]
    demand = demand.rename(columns={'FR': 'demand_FR'})
    demand["year"] = demand.apply(lambda row: int(row["hour"][0:4]), axis=1)
    demand["avg_demand"] = demand.groupby('year')["demand_FR"].transform('mean')
    demand["demand_centered"] = demand["demand_FR"] - demand["avg_demand"]

    # Adding intermittent production information
    pv_france = pv[['hour', 'FR']].rename(columns={'FR': 'pv_FR'})
    onshore_france = onshore[['hour', 'FR']].rename(columns={'FR': 'onshore_FR'})
    offshore_france = offshore[['hour', 'FR']].rename(columns={'FR': 'offshore_FR'})
    river_france = river[['hour', 'FR']].rename(columns={'FR': 'river_FR'})
    river_france["hour"] = river_france.apply(lambda row: row["hour"] + ':00', axis=1)
    vre_profiles = pd.merge(pv_france, onshore_france, on="hour")
    vre_profiles = pd.merge(vre_profiles, offshore_france, on="hour")
    vre_profiles = pd.merge(vre_profiles, river_france, on="hour")
    vre_demand = pd.merge(vre_profiles, demand, on='hour')

    # Adding hydro (PHS and lake) profile (and eventually imports)
    if not imports:
        lake_phs_production = production[["date", "lake_phs", "phs_in"]]
        lake_phs_production = lake_phs_production.rename(columns={'date': 'hour'})
        lake_phs_production['hydro_prod_FR'] = lake_phs_production["lake_phs"] + lake_phs_production["phs_in"]
    else:  # in this case, we take into account net imports as well
        # TODO: je n'ai pas changé les noms par simplicité, mais il faudra changer en "flex" à terme si on confirme que
        # TODO: c'est ok de faire comme ça
        lake_phs_production = production[["date", "lake_phs", "phs_in", "net_imports"]]
        lake_phs_production = lake_phs_production.rename(columns={'date': 'hour'})
        lake_phs_production['hydro_prod_FR'] = lake_phs_production["lake_phs"] + lake_phs_production["phs_in"] + \
                                               lake_phs_production["net_imports"]
    vre_demand = pd.merge(vre_demand, lake_phs_production[["hour", "hydro_prod_FR"]], on="hour")

    # Modify date parameter
    vre_demand["date"] = vre_demand.apply(lambda row: datetime.datetime.strptime(row["hour"], '%Y-%m-%d %H:%M:%S'),
                                          axis=1)
    vre_demand["date_without_year"] = vre_demand.apply(
        lambda row: datetime.datetime.strftime(row["date"], "%m-%d %H-%M-%S"),
        axis=1)
    return vre_demand


def process_hourly_data_merge(pv, onshore, offshore, average_values, labels, bins, save=False, subdir=""):
    """Same as process_hourly_data but here we have to merge two different dataframes because we lack demand data
    for all weather years"""
    if save:
        assert subdir != "", "A string value should be given as input to subdir"
    assert ".csv" not in subdir, "Subdir string should not contain .csv, already included in code."
    # All data year
    pv_france = pv[['hour', 'FR']].rename(columns={'FR': 'pv_FR'})
    onshore_france = onshore[['hour', 'FR']].rename(columns={'FR': 'onshore_FR'})
    offshore_france = offshore[['hour', 'FR']].rename(columns={'FR': 'offshore_FR'})

    # Discretize new weather years
    pv_france['pv_group'] = pd.cut(pv_france['pv_FR'], bins=bins["pv"],
                                   labels=labels[
                                       "pv"])  # we use the same bins and quantiles as the ones defined for period 2015-2019
    onshore_france['onshore_group'] = pd.cut(onshore_france['onshore_FR'], bins=bins["onshore"],
                                             labels=labels["onshore"])
    offshore_france['offshore_group'] = pd.cut(offshore_france['offshore_FR'], bins=bins["offshore"],
                                               labels=labels["offshore"])

    vre_profiles = pd.merge(pv_france, onshore_france, on="hour")
    vre_profiles = pd.merge(vre_profiles, offshore_france, on="hour")
    vre_profiles["year"] = vre_profiles.apply(lambda row: int(row["hour"][0:4]), axis=1)
    vre_profiles["date"] = vre_profiles.apply(
        lambda row: datetime.datetime.strptime(row["hour"], '%Y-%m-%d %H:%M:%S'),
        axis=1)
    vre_profiles["date_without_year"] = vre_profiles.apply(
        lambda row: datetime.datetime.strftime(row["date"], "%m-%d %H-%M-%S"),
        axis=1)

    vre_profiles['pv_group'] = pd.to_numeric(vre_profiles['pv_group'])
    vre_profiles['onshore_group'] = pd.to_numeric(vre_profiles['onshore_group'])
    vre_profiles['offshore_group'] = pd.to_numeric(vre_profiles['offshore_group'])

    # merge with demand and flex data which were averaged over years 2015-2019
    vre_profiles = pd.merge(vre_profiles, average_values, on="date_without_year")
    if save:
        save_dir = "../inputs/" + subdir + ".csv"
        vre_profiles.to_csv(save_dir)
    return vre_profiles


def discretize_distribution(vre_demand, flex="low", save=False, subdir=""):
    """

    :param vre_demand: pd.DataFrame
        DataFrame output from process_hourly_data
    :return:
    """
    assert flex in ["low", "high", "average"]
    if save:
        assert subdir != "", "A string value should be given as input to subdir"  # if we need to save the output
    # We fix the quantiles we are interested in
    pv_quantiles = [0, .5, .6, .65, .7, .75, .8, .9, .95, 1]
    onshore_quantiles = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    demand_quantiles = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .995, 1]
    offshore_quantiles = [0, .3, .6, .9, 1]
    river_quantiles = [0, .3, .6, .9, 1]
    hydro_quantiles = [0, .3, .5, .7, .9, 1]

    out, bins_pv = pd.qcut(vre_demand['pv_FR'], pv_quantiles, retbins=True)
    labels_pv = np.around(bins_pv, decimals=3)[
                :-1]  # we get rid of the final value, and we approximate the number of decimals
    out, bins_onshore = pd.qcut(vre_demand['onshore_FR'], onshore_quantiles, retbins=True)
    labels_onshore = np.around(bins_onshore, decimals=4)[:-1]  # because in practice we find exact values
    out, bins_demand = pd.qcut(vre_demand['demand_centered'], demand_quantiles, retbins=True)
    labels_demand_centered = np.around(bins_demand, decimals=3)[:-1]
    out, bins_offshore = pd.qcut(vre_demand['offshore_FR'], offshore_quantiles, retbins=True)
    labels_offshore = bins_offshore[:-1]  # no need for rounding
    out, bins_river = pd.qcut(vre_demand['river_FR'], river_quantiles, retbins=True)
    labels_river = np.around(bins_river, decimals=3)[:-1]
    out, bins_hydro = pd.qcut(vre_demand['hydro_prod_FR'], hydro_quantiles, retbins=True)
    if flex == "low":
        labels_hydro = np.around(bins_hydro, decimals=3)[:-1]  # initial version
    elif flex == "high":  # we take the other end of the bin
        labels_hydro = np.around(bins_hydro, decimals=3)[1:]
    else:  # flex = "average"
        bins_hydro = np.around(bins_hydro, decimals=3)
        labels_hydro = np.array(
            [(bins_hydro[i] + bins_hydro[i + 1]) / 2 for i in range(bins_hydro.shape[0] - 1)])  # we take the average

    labels = {'pv': labels_pv, 'onshore': labels_onshore, 'demand': labels_demand_centered, 'offshore': labels_offshore,
              'river': labels_river, 'hydro': labels_hydro}

    # We create associated bins
    bins_pv = list(labels_pv[1:])
    bins_pv.append(1)  # we add the maximum value for the pv
    bins_pv.insert(0, -0.001)
    bins_onshore = list(labels_onshore[1:])
    bins_onshore.append(1)  # we add the maximum value for the pv
    bins_onshore.insert(0, -0.001)
    bins_offshore = list(labels_offshore[1:])
    bins_offshore.append(1)  # we add the maximum value for the pv
    bins_offshore.insert(0, -0.001)
    bins = {'pv': bins_pv, 'onshore': bins_onshore, 'offshore': bins_offshore}

    vre_demand['pv_group'] = pd.qcut(vre_demand['pv_FR'], pv_quantiles, labels=labels_pv)
    vre_demand['onshore_group'] = pd.qcut(vre_demand['onshore_FR'], onshore_quantiles, labels=labels_onshore)
    vre_demand['demand_centered_group'] = pd.qcut(vre_demand['demand_centered'],
                                                  demand_quantiles, labels=labels_demand_centered)
    vre_demand['offshore_group'] = pd.qcut(vre_demand['offshore_FR'], offshore_quantiles, labels=labels_offshore)
    vre_demand['river_group'] = pd.qcut(vre_demand['river_FR'], river_quantiles, labels=labels_river)
    vre_demand['hydro_prod_group'] = pd.qcut(vre_demand['hydro_prod_FR'], hydro_quantiles, labels=labels_hydro)

    vre_demand['pv_group'] = pd.to_numeric(vre_demand['pv_group'])
    vre_demand['onshore_group'] = pd.to_numeric(vre_demand['onshore_group'])
    vre_demand['demand_centered_group'] = pd.to_numeric(vre_demand['demand_centered_group'])
    vre_demand['offshore_group'] = pd.to_numeric(vre_demand['offshore_group'])
    vre_demand['river_group'] = pd.to_numeric(vre_demand['river_group'])
    vre_demand['hydro_prod_group'] = pd.to_numeric(vre_demand['hydro_prod_group'])
    if save:
        save_dir = "../inputs/" + subdir + ".csv"
        vre_demand.to_csv(save_dir)
    return vre_demand, labels, bins


def process_average_value(vre_demand, labels):
    """Calculates average value for a given time in year, over multiple years"""
    demand_quantiles = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .995, 1]
    river_quantiles = [0, .3, .6, .9, 1]
    hydro_quantiles = [0, .3, .5, .7, .9, 1]

    average_values = vre_demand.groupby("date_without_year").mean()  # average for a given time in year based on 5 years
    average_values = average_values.reset_index()
    average_values = average_values[["date_without_year", "demand_centered", "river_FR", "hydro_prod_FR"]]

    # we only apply the qcut AFTER we calculated the mean (otherwise we have some problems)
    # REMARK: here, I keep the old labels for demand, river, hydro. This is an approximation, although I do not think
    # it has a strong impact
    average_values['demand_centered_group'] = pd.qcut(average_values['demand_centered'],
                                                      demand_quantiles, labels=labels["demand"])
    average_values['river_group'] = pd.qcut(average_values['river_FR'], river_quantiles, labels=labels["river"])
    average_values['hydro_prod_group'] = pd.qcut(average_values['hydro_prod_FR'], hydro_quantiles,
                                                 labels=labels["hydro"])

    average_values['demand_centered_group'] = pd.to_numeric(average_values['demand_centered_group'])
    average_values['river_group'] = pd.to_numeric(average_values['river_group'])
    average_values['hydro_prod_group'] = pd.to_numeric(average_values['hydro_prod_group'])
    return average_values


def process_joint_law(vre_demand, save=False, subdir=""):
    if save:
        assert subdir != "", "A string value should be given as input to subdir"
    joint_law = vre_demand[
        ['hour', 'pv_group', 'onshore_group', 'offshore_group', 'river_group', 'hydro_prod_group',
         'demand_centered_group']].groupby(
        ['pv_group', 'onshore_group', 'offshore_group', 'river_group', 'hydro_prod_group',
         'demand_centered_group']).count().reset_index()
    joint_law = joint_law.rename(columns={'hour': 'occurrence'})
    joint_law['probability'] = joint_law['occurrence'] / joint_law['occurrence'].sum()
    # We get rid of rows where the associated probability is zero
    joint_law = joint_law.loc[
        joint_law.probability > 0]  # we exclude the weather setting that were not observed
    if save:
        save_dir = "../inputs/" + subdir + ".csv"
        joint_law.to_csv(save_dir)
    return joint_law


def process_joint_law_per_year(start, end, subdir, save_dir):
    load_dir = '../inputs/' + subdir + '.csv'
    for y in np.arange(start, end + 1, 1):
        print(y)
        vre_profiles_tot = pd.read_csv(load_dir, index_col=0)

        vre_profiles_year = vre_profiles_tot.loc[vre_profiles_tot.year == y]  # select year of interest
        print(vre_profiles_year.shape[0])  # check the size of vre_profiles_year to check no inconsistency
        del vre_profiles_tot  # delete to save memory
        save_dir_year = save_dir + f"{y}_joint_law"
        joint_law_year = process_joint_law(vre_profiles_year, save=True, subdir=save_dir_year)
    print("All years processed.")


if __name__ == '__main__':
    # Data for period 2015-2019
    demand = pd.read_csv('../inputs/demand.csv')  # demand profile
    pv = pd.read_csv('../inputs/pv.csv')  # pv capacity factor
    onshore = pd.read_csv('../inputs/onshore_CU.csv')  # onshore capacity factor
    offshore = pd.read_csv('../inputs/offshore_CU.csv')  # offshore capacity factor
    river = pd.read_csv('../inputs/river.csv')  # river capacity factor
    production = pd.read_csv(
        '../inputs/hydro_production.csv')  # hydro generation/storage profile, obtained by running EOLES on different weather years

    # Initial version without the imports
    vre_demand_no_imports = process_hourly_data(demand, pv, onshore, offshore, river, production, imports=False)
    vre_demand_no_imports, labels_no_imports, bins_no_imports = discretize_distribution(vre_demand_no_imports,
                                                                                        save=True,
                                                                                        subdir="vre_profiles_2015_2019")

    # We include imports/exports here
    vre_demand_imports = process_hourly_data(demand, pv, onshore, offshore, river, production, imports=True)
    vre_demand_imports, labels_imports, bins_imports = discretize_distribution(vre_demand_imports, flex="low",
                                                                               save=True,
                                                                               subdir="vre_profiles_2015_2019_imports_low")

    # average_values = process_average_value(vre_demand_imports, labels_imports)
    #
    # # Joint law for the period 2015-2019
    # joint_law = process_joint_law(vre_demand_imports, save=True, subdir="joint_law_imports_average/joint_law_2015_2019")
    #
    # # Weather data for period 1980-2019
    # pv_tot = pd.read_csv('../inputs/pv_1980_2020.csv')
    # onshore_tot = pd.read_csv('../inputs/onshore_CU_1980_2020.csv')
    # offshore_tot = pd.read_csv('../inputs/offshore_CU_1980_2020.csv')
    # vre_profiles_tot = process_hourly_data_merge(pv_tot, onshore_tot, offshore_tot, average_values, labels_imports,
    #                                              bins_imports, save=True, subdir="vre_profiles_imports_average")
    # joint_law_tot = process_joint_law(vre_profiles_tot, save=True,
    #                                   subdir="joint_law_imports_average/joint_law_tot")
    #
    # Processing joint law for each weather year
    process_joint_law_per_year(2015, 2019, "vre_profiles_2015_2019", save_dir="joint_law/2015_2019/")

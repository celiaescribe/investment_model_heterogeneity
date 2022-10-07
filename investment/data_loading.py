import pandas as pd
import numpy as np


def process_data_capa_vom(data_capacity_and_vom, GJ_MWh):
    dispatchable_prod = pd.DataFrame(data_capacity_and_vom,
                                     columns=['name', 'initial_capa', 'fuel_price_GJ', 'efficiency',
                                              'non_fuel_vOM'])
    dispatchable_prod['fuel_cost'] = (1 / dispatchable_prod['efficiency']) * GJ_MWh * \
                                     dispatchable_prod['fuel_price_GJ']
    dispatchable_prod['vOM'] = dispatchable_prod['fuel_cost'] + dispatchable_prod['non_fuel_vOM']
    dispatchable_prod = dispatchable_prod.sort_values(by='vOM')
    dispatchable_prod['cumulative_capa'] = dispatchable_prod['initial_capa'].cumsum()
    return dispatchable_prod


def data_process_homemade(period, avail_nuclear=1, imports=False, flex="low"):
    """Function similar to data_process_with_scenarios but where I change in a very simple manner the capacities
    to obtain more high prices due to gas and imports.
    IN PROGRESS"""
    #TODO: les valeurs choisies ici devraient être modifiées pour avoir un "vrai" sens...
    assert period in ["short", "large"], f"Parameter period should belong to {['short', 'large']}, instead {period}"
    if period == "short":
        if imports:
            assert flex in ["low", "high", "average"]
            subdir_jointlaw = f"inputs/joint_law_imports_{flex}/joint_law_2015_2019.csv"
        else:
            subdir_jointlaw = f"inputs/joint_law/joint_law_2015_2019.csv"
    else:  # "large"
        if imports:
            assert flex in ["low", "high", "average"]
            subdir_jointlaw = f"inputs/joint_law_imports_{flex}/joint_law_tot.csv"
        else:
            subdir_jointlaw = f"inputs/joint_law/joint_law_tot.csv"
    joint_law = pd.read_csv(subdir_jointlaw, index_col=0)

    # Sobriety scenario
    data_capacity_and_vom_2025_sob = [['nuclear', 33.13 * avail_nuclear, 0.47, 0.33, 9],
                                      ['coal_1G', 3, 2, 0.35, 3.3],
                                      ['flex', 23, 4.5, 0.4, 1.6],  # we add a step with flexibility
                                      ['gas_ccgt1G', 6.1, 5.6, 0.4, 1.6],
                                      ['gas_ccgt2G', 9.5, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 7.5, 5.6, 0.58, 1.6],
                                      ['gas_ocgtSA', 1.03, 5.6, 0.42, 1.6],
                                      ['oil_light', 3, 12.9, 0.35, 1.1]]
    data_capacity_and_vom_2030_sob = [['nuclear', 59.4 * avail_nuclear, 0.47, 0.33, 9],
                                      ['gas_ccgt1G', 3.1 - 0.30, 5.6, 0.4, 1.6],
                                      ['gas_ccgt2G', 3.5 - 0.30, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 4.5 - 0.30, 5.6, 0.58, 1.6],
                                      ['gas_ocgtSA', 1.03 - 0.30, 5.6, 0.42, 1.6],
                                      ['oil_light', 1, 12.9, 0.35, 1.1]]
    data_capacity_and_vom_2035_sob = [['nuclear', 54 * avail_nuclear, 0.47, 0.33, 9],
                                      ['gas_ccgt1G', 3.1 - 0.30 - 1, 5.6, 0.4, 1.6],
                                      ['gas_ccgt2G', 3.5 - 0.30 - 1, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 4.5 - 0.30 - 1, 5.6, 0.58, 1.6],
                                      ['oil_light', 1, 12.9, 0.35, 1.1]]  # coal disappears after 2035
    data_capacity_and_vom_2040_sob = [['nuclear', 49.7 * avail_nuclear, 0.47, 0.33, 9],
                                      ['gas_ccgt2G', 3.5 - 3.5, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 4.5 - 3.1, 5.6, 0.58, 1.6]
                                      ]
    data_capacity_and_vom_2045_sob = [['nuclear', 42 * avail_nuclear, 0.47, 0.33, 9],
                                      ['gas_ccgtSA', 1, 5.6, 0.58, 1.6],
                                      ['gas_renewable', 1, 22, 1, 0],
                                      # we choose values for renewable gas so that final price is 80EUR/MWh
                                      ]
    Q_offshore_sob = [0, 5.2, 12, 19.1, 25]


    # Reference scenario  # TODO: attention, je modifie tout ici pour mettre des valeurs un peu aléatoires
    data_capacity_and_vom_2025_ref = [['nuclear', 33.13 * avail_nuclear, 0.47, 0.33, 9],
                                      ['coal_1G', 3, 2, 0.35, 3.3],
                                      ['flex', 23, 4.5, 0.4, 1.6],  # we add a step with flexibility
                                      ['gas_ccgt1G', 6.1, 5.6, 0.4, 1.6],
                                      ['gas_ccgt2G', 6.5, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 7.5, 5.6, 0.58, 1.6],
                                      ['gas_ocgtSA', 1.03, 5.6, 0.42, 1.6],
                                      ['oil_light', 3, 12.9, 0.35, 1.1]]
    data_capacity_and_vom_2030_ref = [['nuclear', 29.4 * avail_nuclear, 0.47, 0.33, 9],
                                      ['flex', 26, 4.5, 0.4, 1.6],  # we add a step with flexibility
                                      ['gas_ccgt1G', 5.1 - 0.15, 5.6, 0.4, 1.6],
                                      ['gas_ccgt2G', 5.5 - 0.15, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 6.5 - 0.15, 5.6, 0.58, 1.6],
                                      ['gas_ocgtSA', 1.03 - 0.15, 5.6, 0.42, 1.6],
                                      ['oil_light', 1, 12.9, 0.35, 1.1]]
    data_capacity_and_vom_2035_ref = [['nuclear', 24 * avail_nuclear, 0.47, 0.33, 9],
                                      ['flex', 30, 4.5, 0.4, 1.6],  # we add a step with flexibility
                                      ['gas_ccgt1G', 4.1 - 0.15 - 0.8, 5.6, 0.4, 1.6],
                                      ['gas_ccgt2G', 4.5 - 0.15 - 0.8, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 5.5 - 0.15 - 0.8, 5.6, 0.58, 1.6],
                                      ['gas_ocgtSA', 1.03 - 0.15 - 0.8, 5.6, 0.42, 1.6],
                                      ['oil_light', 1, 12.9, 0.35, 1.1]]  # coal disappears after 2035
    data_capacity_and_vom_2040_ref = [['nuclear', 19.7 * avail_nuclear, 0.47, 0.33, 9],
                                      ['flex', 33, 4.5, 0.4, 1.6],  # we add a step with flexibility
                                      ['gas_ccgt1G', 3.1 - 0.15 - 1.9, 5.6, 0.4, 1.6],
                                      ['gas_ccgt2G', 3.5 - 0.15 - 1.9, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 4.5 - 0.15 - 1.9, 5.6, 0.58, 1.6],
                                      ['gas_renewable', 1.6, 22, 1, 0]]
    data_capacity_and_vom_2045_ref = [['nuclear', 12 * avail_nuclear, 0.47, 0.33, 9],
                                      ['flex', 36, 4.5, 0.4, 1.6],  # we add a step with flexibility
                                      ['gas_ccgt2G', 3.5 - 0.15 - 2.1 - 1, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 4.5 - 0.15 - 2.1 - 1, 5.6, 0.58, 1.6],
                                      ['gas_renewable', 6.4, 22, 1, 0]]
    Q_offshore_ref = [0, 5.2, 12, 20.9, 30]

    # Reindustrialisation scenario
    data_capacity_and_vom_2025_high = [['nuclear', 63.13 * avail_nuclear, 0.47, 0.33, 9],
                                       ['coal_1G', 3, 2, 0.35, 3.3],
                                       ['gas_ccgt1G', 3.1, 5.6, 0.4, 1.6],
                                       ['gas_ccgt2G', 3.5, 5.6, 0.48, 1.6],
                                       ['gas_ccgtSA', 4.5, 5.6, 0.58, 1.6],
                                       ['gas_ocgtSA', 1.03, 5.6, 0.42, 1.6],
                                       ['oil_light', 3, 12.9, 0.35, 1.1]]
    data_capacity_and_vom_2030_high = [['nuclear', 59.4 * avail_nuclear, 0.47, 0.33, 9],
                                       ['gas_ccgt1G', 3.1 - 0.15, 5.6, 0.4, 1.6],
                                       ['gas_ccgt2G', 3.5 - 0.15, 5.6, 0.48, 1.6],
                                       ['gas_ccgtSA', 4.5 - 0.15, 5.6, 0.58, 1.6],
                                       ['gas_ocgtSA', 1.03 - 0.15, 5.6, 0.42, 1.6],
                                       ['oil_light', 1, 12.9, 0.35, 1.1]]
    data_capacity_and_vom_2035_high = [['nuclear', 54 * avail_nuclear, 0.47, 0.33, 9],
                                       ['gas_ccgt1G', 3.1 - 0.15 - 0.8, 5.6, 0.4, 1.6],
                                       ['gas_ccgt2G', 3.5 - 0.15 - 0.8, 5.6, 0.48, 1.6],
                                       ['gas_ccgtSA', 4.5 - 0.15 - 0.8, 5.6, 0.58, 1.6],
                                       ['gas_ocgtSA', 1.03 - 0.15 - 0.8, 5.6, 0.42, 1.6],
                                       ['oil_light', 1, 12.9, 0.35, 1.1]]  # coal disappears after 2035
    data_capacity_and_vom_2040_high = [['nuclear', 49.7 * avail_nuclear, 0.47, 0.33, 9],
                                       ['gas_ccgt1G', 3.1 - 0.15 - 1.9, 5.6, 0.4, 1.6],
                                       ['gas_ccgt2G', 3.5 - 0.15 - 1.9, 5.6, 0.48, 1.6],
                                       ['gas_ccgtSA', 4.5 - 0.15 - 1.9, 5.6, 0.58, 1.6],
                                       ['gas_renewable', 1.6, 22, 1, 0]
                                       ]
    data_capacity_and_vom_2045_high = [['nuclear', 42 * avail_nuclear, 0.47, 0.33, 9],
                                       ['gas_ccgt2G', 3.5 - 0.15 - 2.1 - 1, 5.6, 0.48, 1.6],
                                       ['gas_ccgtSA', 4.5 - 0.15 - 2.1 - 1, 5.6, 0.58, 1.6],
                                       ['gas_renewable', 9.7, 22, 1, 0]
                                       ]
    Q_offshore_high = [0, 5, 20, 33.1, 48]

    data_capacity_and_vom_sob = {
        '2025': data_capacity_and_vom_2025_sob,
        '2030': data_capacity_and_vom_2030_sob,
        '2035': data_capacity_and_vom_2035_sob,
        '2040': data_capacity_and_vom_2040_sob,
        '2045': data_capacity_and_vom_2045_sob,
    }

    data_capacity_and_vom_ref = {
        '2025': data_capacity_and_vom_2025_ref,
        '2030': data_capacity_and_vom_2030_ref,
        '2035': data_capacity_and_vom_2035_ref,
        '2040': data_capacity_and_vom_2040_ref,
        '2045': data_capacity_and_vom_2045_ref,
    }

    data_capacity_and_vom_high = {
        '2025': data_capacity_and_vom_2025_high,
        '2030': data_capacity_and_vom_2030_high,
        '2035': data_capacity_and_vom_2035_high,
        '2040': data_capacity_and_vom_2040_high,
        '2045': data_capacity_and_vom_2045_high,
    }

    data_capacity_and_vom = {
        'sob': data_capacity_and_vom_sob,
        'ref': data_capacity_and_vom_ref,
        'high': data_capacity_and_vom_high
    }

    Q_offshore_dict = {
        'sob': Q_offshore_sob,
        'ref': Q_offshore_ref,
        'high': Q_offshore_high
    }

    x_cutoff, y_cutoff, Q_offshore = prepare_cutoff(data_capacity_and_vom, Q_offshore_dict)
    return joint_law, x_cutoff, y_cutoff, Q_offshore


def data_process_with_scenarios(period, avail_nuclear, imports=False, flex="low"):
    """
    Function which returns the cutoff points, joint law, and values for offshore. This is the latest version of the
    function, where we added coupling between demand scenario and energy mix scenario. We also take into account the
    evolution of offshore values which are exogeneous to the scenario.
    We also add a param avail_nuclear which captures the availability of the nuclear fleet.
    Often, this fleet is not available at 100%
    ----------
    imports: bool
        Indicates which joint law should be considered, whether it includes imports flexibility or not
    avail_nuclear: float
        Indicates the availability of the nuclear fleet"""
    assert period in ["short", "large"], f"Parameter period should belong to {['short', 'large']}, instead {period}"
    if period == "short":
        if imports:
            assert flex in ["low", "high", "average"]
            subdir_jointlaw = f"inputs/joint_law_imports_{flex}/joint_law_2015_2019.csv"
        else:
            subdir_jointlaw = f"inputs/joint_law/joint_law_2015_2019.csv"
    else:  # "large"
        if imports:
            assert flex in ["low", "high", "average"]
            subdir_jointlaw = f"inputs/joint_law_imports_{flex}/joint_law_tot.csv"
        else:
            subdir_jointlaw = f"inputs/joint_law/joint_law_tot.csv"
    joint_law = pd.read_csv(subdir_jointlaw, index_col=0)
    # Sobriety scenario
    data_capacity_and_vom_2025_sob = [['nuclear', 63.13 * avail_nuclear, 0.47, 0.33, 9],
                                      ['coal_1G', 3, 2, 0.35, 3.3],
                                      ['gas_ccgt1G', 3.1, 5.6, 0.4, 1.6],
                                      ['gas_ccgt2G', 3.5, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 4.5, 5.6, 0.58, 1.6],
                                      ['gas_ocgtSA', 1.03, 5.6, 0.42, 1.6],
                                      ['oil_light', 3, 12.9, 0.35, 1.1]]
    data_capacity_and_vom_2030_sob = [['nuclear', 59.4 * avail_nuclear, 0.47, 0.33, 9],
                                      ['gas_ccgt1G', 3.1 - 0.30, 5.6, 0.4, 1.6],
                                      ['gas_ccgt2G', 3.5 - 0.30, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 4.5 - 0.30, 5.6, 0.58, 1.6],
                                      ['gas_ocgtSA', 1.03 - 0.30, 5.6, 0.42, 1.6],
                                      ['oil_light', 1, 12.9, 0.35, 1.1]]
    data_capacity_and_vom_2035_sob = [['nuclear', 54 * avail_nuclear, 0.47, 0.33, 9],
                                      ['gas_ccgt1G', 3.1 - 0.30 - 1, 5.6, 0.4, 1.6],
                                      ['gas_ccgt2G', 3.5 - 0.30 - 1, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 4.5 - 0.30 - 1, 5.6, 0.58, 1.6],
                                      ['oil_light', 1, 12.9, 0.35, 1.1]]  # coal disappears after 2035
    data_capacity_and_vom_2040_sob = [['nuclear', 49.7 * avail_nuclear, 0.47, 0.33, 9],
                                      ['gas_ccgt2G', 3.5 - 3.5, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 4.5 - 3.1, 5.6, 0.58, 1.6]
                                      ]
    data_capacity_and_vom_2045_sob = [['nuclear', 42 * avail_nuclear, 0.47, 0.33, 9],
                                      ['gas_ccgtSA', 1, 5.6, 0.58, 1.6],
                                      ['gas_renewable', 1, 22, 1, 0],
                                      # we choose values for renewable gas so that final price is 80EUR/MWh
                                      ]
    Q_offshore_sob = [0, 5.2, 12, 19.1, 25]

    # Reference scenario
    data_capacity_and_vom_2025_ref = [['nuclear', 63.13 * avail_nuclear, 0.47, 0.33, 9],
                                      ['coal_1G', 3, 2, 0.35, 3.3],
                                      ['gas_ccgt1G', 3.1, 5.6, 0.4, 1.6],
                                      ['gas_ccgt2G', 3.5, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 4.5, 5.6, 0.58, 1.6],
                                      ['gas_ocgtSA', 1.03, 5.6, 0.42, 1.6],
                                      ['oil_light', 3, 12.9, 0.35, 1.1]]
    data_capacity_and_vom_2030_ref = [['nuclear', 59.4 * avail_nuclear, 0.47, 0.33, 9],
                                      ['gas_ccgt1G', 3.1 - 0.15, 5.6, 0.4, 1.6],
                                      ['gas_ccgt2G', 3.5 - 0.15, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 4.5 - 0.15, 5.6, 0.58, 1.6],
                                      ['gas_ocgtSA', 1.03 - 0.15, 5.6, 0.42, 1.6],
                                      ['oil_light', 1, 12.9, 0.35, 1.1]]
    data_capacity_and_vom_2035_ref = [['nuclear', 54 * avail_nuclear, 0.47, 0.33, 9],
                                      ['gas_ccgt1G', 3.1 - 0.15 - 0.8, 5.6, 0.4, 1.6],
                                      ['gas_ccgt2G', 3.5 - 0.15 - 0.8, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 4.5 - 0.15 - 0.8, 5.6, 0.58, 1.6],
                                      ['gas_ocgtSA', 1.03 - 0.15 - 0.8, 5.6, 0.42, 1.6],
                                      ['oil_light', 1, 12.9, 0.35, 1.1]]  # coal disappears after 2035
    data_capacity_and_vom_2040_ref = [['nuclear', 49.7 * avail_nuclear, 0.47, 0.33, 9],
                                      ['gas_ccgt1G', 3.1 - 0.15 - 1.9, 5.6, 0.4, 1.6],
                                      ['gas_ccgt2G', 3.5 - 0.15 - 1.9, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 4.5 - 0.15 - 1.9, 5.6, 0.58, 1.6],
                                      ['gas_renewable', 1.6, 22, 1, 0]]
    data_capacity_and_vom_2045_ref = [['nuclear', 42 * avail_nuclear, 0.47, 0.33, 9],
                                      ['gas_ccgt2G', 3.5 - 0.15 - 2.1 - 1, 5.6, 0.48, 1.6],
                                      ['gas_ccgtSA', 4.5 - 0.15 - 2.1 - 1, 5.6, 0.58, 1.6],
                                      ['gas_renewable', 6.4, 22, 1, 0]]
    Q_offshore_ref = [0, 5.2, 12, 20.9, 30]

    # Reindustrialisation scenario
    data_capacity_and_vom_2025_high = [['nuclear', 63.13 * avail_nuclear, 0.47, 0.33, 9],
                                       ['coal_1G', 3, 2, 0.35, 3.3],
                                       ['gas_ccgt1G', 3.1, 5.6, 0.4, 1.6],
                                       ['gas_ccgt2G', 3.5, 5.6, 0.48, 1.6],
                                       ['gas_ccgtSA', 4.5, 5.6, 0.58, 1.6],
                                       ['gas_ocgtSA', 1.03, 5.6, 0.42, 1.6],
                                       ['oil_light', 3, 12.9, 0.35, 1.1]]
    data_capacity_and_vom_2030_high = [['nuclear', 59.4 * avail_nuclear, 0.47, 0.33, 9],
                                       ['gas_ccgt1G', 3.1 - 0.15, 5.6, 0.4, 1.6],
                                       ['gas_ccgt2G', 3.5 - 0.15, 5.6, 0.48, 1.6],
                                       ['gas_ccgtSA', 4.5 - 0.15, 5.6, 0.58, 1.6],
                                       ['gas_ocgtSA', 1.03 - 0.15, 5.6, 0.42, 1.6],
                                       ['oil_light', 1, 12.9, 0.35, 1.1]]
    data_capacity_and_vom_2035_high = [['nuclear', 54 * avail_nuclear, 0.47, 0.33, 9],
                                       ['gas_ccgt1G', 3.1 - 0.15 - 0.8, 5.6, 0.4, 1.6],
                                       ['gas_ccgt2G', 3.5 - 0.15 - 0.8, 5.6, 0.48, 1.6],
                                       ['gas_ccgtSA', 4.5 - 0.15 - 0.8, 5.6, 0.58, 1.6],
                                       ['gas_ocgtSA', 1.03 - 0.15 - 0.8, 5.6, 0.42, 1.6],
                                       ['oil_light', 1, 12.9, 0.35, 1.1]]  # coal disappears after 2035
    data_capacity_and_vom_2040_high = [['nuclear', 49.7 * avail_nuclear, 0.47, 0.33, 9],
                                       ['gas_ccgt1G', 3.1 - 0.15 - 1.9, 5.6, 0.4, 1.6],
                                       ['gas_ccgt2G', 3.5 - 0.15 - 1.9, 5.6, 0.48, 1.6],
                                       ['gas_ccgtSA', 4.5 - 0.15 - 1.9, 5.6, 0.58, 1.6],
                                       ['gas_renewable', 1.6, 22, 1, 0]
                                       ]
    data_capacity_and_vom_2045_high = [['nuclear', 42 * avail_nuclear, 0.47, 0.33, 9],
                                       ['gas_ccgt2G', 3.5 - 0.15 - 2.1 - 1, 5.6, 0.48, 1.6],
                                       ['gas_ccgtSA', 4.5 - 0.15 - 2.1 - 1, 5.6, 0.58, 1.6],
                                       ['gas_renewable', 9.7, 22, 1, 0]
                                       ]
    Q_offshore_high = [0, 5, 20, 33.1, 48]

    data_capacity_and_vom_sob = {
        '2025': data_capacity_and_vom_2025_sob,
        '2030': data_capacity_and_vom_2030_sob,
        '2035': data_capacity_and_vom_2035_sob,
        '2040': data_capacity_and_vom_2040_sob,
        '2045': data_capacity_and_vom_2045_sob,
    }

    data_capacity_and_vom_ref = {
        '2025': data_capacity_and_vom_2025_ref,
        '2030': data_capacity_and_vom_2030_ref,
        '2035': data_capacity_and_vom_2035_ref,
        '2040': data_capacity_and_vom_2040_ref,
        '2045': data_capacity_and_vom_2045_ref,
    }

    data_capacity_and_vom_high = {
        '2025': data_capacity_and_vom_2025_high,
        '2030': data_capacity_and_vom_2030_high,
        '2035': data_capacity_and_vom_2035_high,
        '2040': data_capacity_and_vom_2040_high,
        '2045': data_capacity_and_vom_2045_high,
    }

    data_capacity_and_vom = {
        'sob': data_capacity_and_vom_sob,
        'ref': data_capacity_and_vom_ref,
        'high': data_capacity_and_vom_high
    }

    Q_offshore_dict = {
        'sob': Q_offshore_sob,
        'ref': Q_offshore_ref,
        'high': Q_offshore_high
    }

    x_cutoff, y_cutoff, Q_offshore = prepare_cutoff(data_capacity_and_vom, Q_offshore_dict)
    return joint_law, x_cutoff, y_cutoff, Q_offshore


def prepare_cutoff(data_capacity_and_vom, Q_offshore_dict):
    GJ_MWh = 3.6  # conversion factor
    x_cutoff = []
    y_cutoff = []
    Q_offshore = []
    t = 0
    for year in ['2025', '2030', '2035', '2040', '2045']:
        x_cutoff_year = []
        y_cutoff_year = []
        Q_offshore_year = []
        for scenario in ["sob", "ref", "high"]:
            Q_offshore_year.append(Q_offshore_dict[scenario][t])
            data_capacity_and_vom_year_scenario = data_capacity_and_vom[scenario][year]
            dispatchable_prod_year = process_data_capa_vom(data_capacity_and_vom_year_scenario, GJ_MWh)

            # define the cutoff points necessary for supply function
            x_cutoff_year_scenario = list(dispatchable_prod_year.cumulative_capa)
            x_cutoff_year_scenario.insert(0, 0)
            x_cutoff_year.append(x_cutoff_year_scenario)
            # x_cutoff[year] = x_cutoff_year

            y_cutoff_year_scenario = list(dispatchable_prod_year.vOM)
            y_cutoff_year_scenario.insert(0, 0)
            y_cutoff_year.append(y_cutoff_year_scenario)
            # y_cutoff[year] = y_cutoff_year
        x_cutoff.append(x_cutoff_year)
        y_cutoff.append(y_cutoff_year)
        Q_offshore.append(Q_offshore_year)
        t += 1  # update the cursor indicating the considered year
    return x_cutoff, y_cutoff, Q_offshore


def load_all_weather_data(start, end, imports=False, flex="low"):
    """Returns numpy arrays where all possible weather samples and associated probability are present, for each year
    included between start and end included. The vector years_tot indicate which year corresponds to which samples.
     It should be noted that the same sample could be present multiple times, with different probability,
     corresponding to different years.
         ----------
    start: int
        Start year to consider
    end: int
        End year to consider
    imports: bool
        Indicates which joint law should be considered, whether it includes imports flexibility or not
    """
    demand_centered_tot = np.array([])
    pv_tot = np.array([])
    onshore_tot = np.array([])
    offshore_tot = np.array([])
    river_tot = np.array([])
    hydro_prod_tot = np.array([])
    probability_tot = np.array([])
    year_tot = np.array([])
    for y in np.arange(start, end + 1, 1):
        if imports:
            assert flex in ["low", "high", "average"], "Parameter flex should be equal to low, high or average"
            subdir_jointlaw = f"inputs/joint_law_imports_{flex}/{y}_joint_law.csv"

        else:
            subdir_jointlaw = f"inputs/joint_law/{y}_joint_law.csv"

        joint_law = pd.read_csv(subdir_jointlaw, index_col=0)  # loading joint law for given year

        demand_centered = np.array(joint_law.demand_centered_group)
        pv = np.array(joint_law.pv_group)
        onshore = np.array(joint_law.onshore_group)
        offshore = np.array(joint_law.offshore_group)
        river = np.array(joint_law.river_group)
        hydro_prod = np.array(joint_law.hydro_prod_group)
        probability = np.array(joint_law.probability)
        year = np.full(probability.shape, y)

        demand_centered_tot = np.concatenate((demand_centered_tot, demand_centered))
        pv_tot = np.concatenate((pv_tot, pv))
        onshore_tot = np.concatenate((onshore_tot, onshore))
        offshore_tot = np.concatenate((offshore_tot, offshore))
        river_tot = np.concatenate((river_tot, river))
        hydro_prod_tot = np.concatenate((hydro_prod_tot, hydro_prod))
        probability_tot = np.concatenate((probability_tot, probability))
        year_tot = np.concatenate((year_tot, year))
    return demand_centered_tot, pv_tot, onshore_tot, offshore_tot, river_tot, hydro_prod_tot, probability_tot, year_tot

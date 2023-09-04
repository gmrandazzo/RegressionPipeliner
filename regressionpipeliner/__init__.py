"""
__init__.py

Copyright (C) <2023>  Giuseppe Marco Randazzo <gmrandazzo@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import sys
import os
import random
import logging
from copy import copy
import numpy as np
from codecarbon import track_emissions
from codecarbon import OfflineEmissionsTracker
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from .dataset import (
    Split,
    make_split
)

from .boostedtrees import(
    xgbreg,
    catbreg
)

from .linear import(
    pls,
    pls_sklearn
)
from .dnn import tf_dnn

from .io import elaborate_results

def regress(split : Split):
    """
    Regress using sklearn and tensorflow models
    """
    print(">> Regress ")
    names = [
        "Linear-SVR",
        "RBF-SVR",
        "DecisionTree",
        "ExtraTrees",
        "RandomForest",
        "KNeighbors"
    ]

    common_params = {'n_estimators':500,
                     'max_depth':5,
                     'min_samples_leaf':9,
                     'min_samples_split':9,
                     'random_state':0
                    }

    regressor = [
        SVR(kernel="linear", C=0.025),
        SVR(kernel="rbf", gamma="scale", C=1),
        DecisionTreeRegressor(max_depth=5, random_state=0),
        ExtraTreesRegressor(**common_params),
        RandomForestRegressor(**common_params),
        KNeighborsRegressor(n_neighbors=5)
    ]

    # In this case we do not use the test set for algorithm reasons.
    regress_results = {}
    emissions_results = {}

    for name, clf in zip(names, regressor):
        logging.info(" * Calculating %s", (name))
        tracker = OfflineEmissionsTracker(country_iso_code="CHE")
        tracker.start()
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(split.x_train, split.y_train)
        regress_results[name] = clf.predict(split.x_val)
        emissions_results[name] = float(tracker.stop())

    logging.info(" * Calculating %s", ('PLS'))
    tracker = OfflineEmissionsTracker(country_iso_code="CHE")
    tracker.start()
    regress_results["PLS"] = pls(split)
    emissions_results["PLS"] = float(tracker.stop())

    logging.info(" * Calculating %s", ('PLS sklearn'))
    tracker = OfflineEmissionsTracker(country_iso_code="CHE")
    tracker.start()
    regress_results["PLS-SKLEARN"] = pls_sklearn(split)
    emissions_results["PLS-SKLEARN"] = float(tracker.stop())

    logging.info(" * Calculating %s", ('XGBoost'))
    tracker = OfflineEmissionsTracker(country_iso_code="CHE")
    tracker.start()
    regress_results["XGBoost"] = xgbreg(split)
    emissions_results["XGBoost"] = float(tracker.stop())

    logging.info(" * Calculating %s", ('CatBoost'))
    tracker = OfflineEmissionsTracker(country_iso_code="CHE")
    tracker.start()
    regress_results["CatBoost"] = catbreg(split)
    emissions_results["CatBoost"] = float(tracker.stop())

    logging.info(" * Calculating %s", ('DNN'))
    tracker = OfflineEmissionsTracker(country_iso_code="CHE")
    tracker.start()
    regress_results["DNN"] = tf_dnn(split)
    emissions_results["DNN"] = float(tracker.stop())
    return regress_results, emissions_results

def best_regressor(xdict : dict,
                   x_header : list,
                   ydict : dict,
                   split_fnc=make_split):
    """
    Find the best regressor
    """
    print(">> Search for best regressor")
    split = split_fnc(xdict, x_header, ydict, 2785)
    regress_results, emissions_results = regress(split)
    return elaborate_results(split.y_val, regress_results, emissions_results), split

def variable_importance(split : Split,
                        res_all_vars : dict):
    """
    Variable importance
    This take times...
    We build N models with N the number of variable and we see how
    the performance decrease while killing a varialbe.
    We define a score S which represent the variable importance
    as s_current/s_original.
    Values > 1 means killing this variable is better
    Values < 1 means this variable is important to explain the model
    """
    logging.info(" * Calculate the variable importance")
    v_imp = {}
    for j, var_name in enumerate(split.x_header):
        var_name = split.x_header[j]
        print(split_copy.x_train.shape)
        logging.info("    - Kill %s -> Train shape %d %d", var_name,
                                                            split_copy.x_train.shape[0],
                                                            split_copy.x_train.shape[1])
        split_copy = copy(split)
        split_copy.x_train = np.delete(split_copy.x_train, j, 1)
        split_copy.x_test = np.delete(split_copy.x_test, j, 1)
        split_copy.x_val = np.delete(split_copy.x_val, j, 1)
        regress_results = regress(split_copy)
        curr_res = elaborate_results(split_copy.y_val,
                                     regress_results,
                                     None)
        v_imp[var_name] = {}
        for key in res_all_vars.keys():
            v_imp[var_name][key] = curr_res[key]/res_all_vars[key]
    return v_imp

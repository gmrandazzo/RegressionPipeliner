"""
linear.py

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
import numpy as np
from libscientific.pls import PLS
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from .dataset import Split


def pls(split : Split,
        nlv = 50):
    """
    Partial Least Squares (PLS) using the NIPALS algorithm

    Parameters:
        - split (Split): An instance of the `Split` class containing
          training, testing, and validation data splits.
        - nlv (int): The number of latent variables to use for PLS regression (default is 50).

    Returns:
        - p_y_val (List[float]): A list of predicted target values for the validation dataset.

    Usage:
        This function takes a `Split` object as input, containing the necessary data splits
        for training, testing, and validation. It performs PLS regression with the specified
        number of latent variables (nlv) and returns a list of predicted target values for
        the validation dataset. This function is useful for predictive modeling tasks where
        dimensionality reduction and prediction of continuous target variables are required.

    Example:
        # Import the necessary modules and create a Split object
        from regressionpipeliner import pls, Split

        data_split = Split()
        data_split.x_train = training_features
        data_split.y_train = training_labels
        data_split.x_test = testing_features
        data_split.y_test = testing_labels
        data_split.x_val = validation_features
        data_split.y_val = validation_labels

        # Perform PLS regression
        predicted_values = pls(data_split, nlv=30)
    """
    clf = PLS(nlv=nlv, xscaling=1, yscaling=0)
    y_train = [[item] for item in split.y_train]
    clf.fit(split.x_train, y_train)
    p_c_y_test, _ = clf.predict(split.x_test)
    # Select the best latent variables
    res = []
    for j in range(len(p_c_y_test[0])):
        ypred = []
        for _, row in enumerate(p_c_y_test):
            ypred.append(row[j])
        res.append(r2_score(split.y_test, ypred))
    latent_variables = 0
    for i in range(1, len(res)):
        if res[i] > res[i-1] and np.abs(res[i]-res[i-1]+res[i-1]) > 0.01:
            latent_variables += 1
        else:
            break
    latent_variables = np.argmax(res)
    p_c_y_val, _ = clf.predict(split.x_val)
    p_y_val = []
    for i, row in enumerate(p_c_y_val):
        p_y_val.append(row[latent_variables])
    return p_y_val

def pls_sklearn(split : Split,
                nlv_=50):
    """
    Partial Least Squares (PLS) using scikit-learn method

    Parameters:
        - split (Split): An instance of the `Split` class containing
          training, testing, and validation data splits.
        - nlv (int): The number of latent variables to use for PLS regression (default is 50).

    Returns:
        - p_y_val (List[float]): A list of predicted target values for the validation dataset.

    Usage:
        This function takes a `Split` object as input, containing the necessary data splits for
        raining, testing, and validation. It performs PLS regression with the specified number
        of latent variables (nlv) and returns a list of predicted target values for
        the validation dataset.
        This function is useful for predictive modeling tasks where dimensionality reduction and
        prediction of continuous target variables are required.

    Example:
        # Import the necessary modules and create a Split object
        from regressionpipeliner import pls, Split

        data_split = Split()
        data_split.x_train = training_features
        data_split.y_train = training_labels
        data_split.x_test = testing_features
        data_split.y_test = testing_labels
        data_split.x_val = validation_features
        data_split.y_val = validation_labels

        # Perform PLS regression
        predicted_values = pls_sklearn(data_split, nlv=30)
    """
    scaler = StandardScaler()
    scaler.fit(split.x_train)
    y_pred = [[0 for j in range(nlv_)] for i in range(len(split.x_test))]
    for nlv in range(1, nlv_):
        clf = PLSRegression(n_components=nlv)
        clf.fit(scaler.transform(split.x_train), split.y_train)
        ypred = clf.predict(scaler.transform(split.x_test))
        for j, val in enumerate(ypred):
            y_pred[j][nlv-1] = val
        del clf
    res = []
    for j in range(len(y_pred[0])):
        ypred = []
        for _, row in enumerate(y_pred):
            ypred.append(row[j])
        res.append(r2_score(split.y_test, ypred))
    latent_variables = 0
    for i in range(1, len(res)):
        if res[i] > res[i-1] and np.abs(res[i]-res[i-1]+res[i-1]) > 0.01:
            latent_variables += 1
        else:
            break
    latent_variables = np.argmax(res)
    if latent_variables == 0:
        latent_variables = 1
    clf = PLSRegression(n_components=latent_variables)
    clf.fit(scaler.transform(split.x_train), split.y_train)
    return clf.predict(scaler.transform(split.x_val))

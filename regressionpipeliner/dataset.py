"""
dataset.py

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
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from .prep import descriptors_preprocess


class Split():
    """
    Class that defines a split for training, testing, and validation datasets.

    Attributes:
        x_header (str): A description or label for the input features.
        x_train (array-like): The training data for input features.
        y_train (array-like): The training data for output labels or targets.
        x_test (array-like): The testing data for input features.
        y_test (array-like): The testing data for output labels or targets.
        x_val (array-like): The validation data for input features.
        y_val (array-like): The validation data for output labels or targets.

    Methods:
        __repr__(): Returns a string representation of the class name.
        __str__(): Returns a string representation of the class name.

    Usage:
        This class is intended to store data splits for machine learning tasks, such as
        training, testing, and validation data. Users can set the attributes to store
        the respective data splits for later use in modeling and evaluation.

    Example:
        # Create a Split object and assign data splits
        data_split = Split()
        data_split.x_header = "Feature Descriptions"
        data_split.x_train = training_features
        data_split.y_train = training_labels
        data_split.x_test = testing_features
        data_split.y_test = testing_labels
        data_split.x_val = validation_features
        data_split.y_val = validation_labels

        # Access the stored data splits
        print(data_split.x_header)  # Print the feature description
        print(len(data_split.x_train))  # Print the number of training samples
    """
    def __init__(self):
        self.x_header : list = None
        self.x_train : np.array = None
        self.y_train : np.array = None
        self.x_test : np.array = None
        self.y_test : np.array = None
        self.x_val : np.array = None
        self.y_val : np.array = None

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__


def make_sample(xdict, ydict, keys):
    """
    Create a sample given a list of keys.

    Parameters:
        xdict (dict): A dictionary containing input features as values associated
                      with object names as keys.
        ydict (dict): A dictionary containing output labels or targets as values associated
                      with object names as keys.
        keys (list, optional): A list of keys to select specific samples (default is None).

    Returns:
        numpy.ndarray: A 2D numpy array containing input features.
        numpy.ndarray: A 1D numpy array containing output labels or targets.

    This function creates a sample by selecting input features and output labels based on a list
    of keys. If `keys` is not provided, it selects all samples available
    in both `xdict` and `ydict`.
    The function returns numpy arrays containing input features and output labels.
    """
    logging.info(" * Create Sample out of a list of keys")
    x_mx = []
    y_mx = []
    if keys is None:
        for key in ydict.keys():
            if key in xdict.keys():
                x_mx.append(xdict[key])
                y_mx.append(ydict[key])
            else:
                continue
    else:
        for key in keys:
            if key in xdict.keys() and key in ydict.keys():
                x_mx.append(xdict[key])
                y_mx.append(ydict[key])
            else:
                continue
    return np.array(x_mx, dtype=float), np.array(y_mx, dtype=float)


def make_split(xdict : dict,
               x_header : list,
               ydict : dict,
               random_state=2785):
    """
    Create a split of train/test/validation data for comparing regressors.

    Parameters:
        xdict (dict): A dictionary containing input features as values associated
                      with object names as keys.
        x_header (list): A list of feature names corresponding to the input features.
        ydict (dict): A dictionary containing output labels or targets as values associated
                      with object names as keys.
        random_state (int, optional): Random seed for reproducibility (default is 2785).

    Returns:
        Split: A `Split` object containing train, test, and validation data splits.

    This function creates a random split of the data into training, testing, and validation sets.
    It returns a `Split` object containing the data splits. It also preprocesses the input features
    by removing less important descriptors and updates the `x_header` accordingly.
    """
    logging.info(" * Create a random Train/Test/Validation split")
    split = Split()
    keys = list(ydict.keys())
    k_sub, k_val = train_test_split(keys,
                                    test_size=0.33,
                                    random_state=random_state)
    k_train, k_test = train_test_split(k_sub,
                                       test_size=0.2,
                                       random_state=random_state)

    split.x_train, split.y_train = make_sample(xdict, ydict, k_train)
    split.x_test, split.y_test = make_sample(xdict, ydict, k_test)
    split.x_val, split.y_val = make_sample(xdict, ydict, k_val)

    # Preprocess descriptors by removing the less important descriptors
    x_all, _ = make_sample(xdict, ydict, None)
    _, skip_ids = descriptors_preprocess(x_all, None)
    split.x_train, _ = descriptors_preprocess(split.x_train, skip_ids)
    split.x_test, _ = descriptors_preprocess(split.x_test, skip_ids)
    split.x_val, _ = descriptors_preprocess(split.x_val, skip_ids)
    split.x_header = []
    for i, h_name in enumerate(x_header):
        if i not in skip_ids:
            split.x_header.append(h_name)
    return split

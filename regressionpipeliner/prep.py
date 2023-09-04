#!/usr/bin/env python3
"""
prep.py

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

def get_low_variance_descriptors(x_mx):
    """
    Get low variance descriptors.

    Parameters:
        x_mx (numpy.ndarray): A 2D numpy array containing descriptors.

    Returns:
        list: A list of column indices to skip, representing low variance descriptors.

    This function calculates the standard deviation of each descriptor column in the input
    array `x_mx` and identifies columns with low variance. Columns with a standard deviation
    less than 1% of the range of values are considered low variance and are added to the list
    of indices to skip.
    """
    logging.info(" * Get low variance descriptors")
    skip_ids = []
    for j in range(len(x_mx[0])):
        vect = []
        for i, _ in enumerate(x_mx):
            if np.isnan(x_mx[i][j]):
                skip_ids.append(j)
                break
            vect.append(x_mx[i][j])
        try:
            column_var = np.var(vect)
            if column_var < 0.01:
                skip_ids.append(j)
        except ValueError as err:
            logging.error(err)
            skip_ids.append(j)
    return skip_ids


def descriptors_preprocess(x_mx, skip_ids=None):
    """
    Preprocess descriptors by removing not relevant columns.

    Parameters:
        x_mx (numpy.ndarray): A 2D numpy array containing descriptors.
        skip_ids (list, optional): A list of column indices to skip (default is None).

    Returns:
        numpy.ndarray: A preprocessed numpy array containing relevant descriptors.
        list: A list of column indices that were skipped.

    This function preprocesses the input descriptor matrix `x_mx` by removing columns
    that are not relevant for analysis. If `skip_ids` is not provided, it calculates and
    uses the low variance descriptor indices obtained from the `get_low_variance_descriptors`
    function. It returns the preprocessed descriptor matrix and a list of skipped column
    indices.
    """
    if skip_ids is None:
        skip_lst = get_low_variance_descriptors(x_mx)
    else:
        skip_lst = skip_ids

    logging.info(" * Preprocess descriptors: remove nan and low variance descs.")
    x_pre = []
    for _, row in enumerate(x_mx):
        x_pre.append([])
        for j, val in enumerate(row):
            if j not in skip_lst:
                x_pre[-1].append(val)
    return np.array(x_pre, dtype=float), skip_lst

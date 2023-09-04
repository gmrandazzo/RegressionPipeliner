#!/usr/bin/env python3
"""
make_model.py

Copyright (C) <2022>  Giuseppe Marco Randazzo <gmrandazzo@gmail.com>

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

Algorhtm:

Run several classifiers also a neural network and get the best model


"""

import sys
import csv
import os
from scipy.signal import savgol_filter
import numpy as np
from copy import copy
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../" % (dir_path))
from basic_methods import Split
from basic_methods import descriptors_preprocess
from basic_methods import make_sample
from basic_methods import best_regressor
from basic_methods import write_results


def readinput(f_inp):
    split_dict = {}
    x_dict = {}
    y_dict = {}
    x_header = None
    y_header = None
    with open(f_inp) as f:
        r = csv.reader(f, delimiter=',', quotechar='"')
        i = 0
        for row in r:
            if row[0] == "Set":
                y_header = [row[8]]
                x_header = row[9:]
            else:
                key = "OBJ%d" % (i)
                set_type = row[0].strip()
                if set_type in split_dict.keys():
                    split_dict[set_type].append(key)
                else:
                    split_dict[set_type] = [key]
                y_dict[key] = row[8]
                x_dict[key] = row[9:]
                i+=1
    return x_dict, x_header, y_dict, y_header, split_dict


def mx_join(arr1, arr2):
    assert len(arr1) == len(arr2), "Objects in array of different size"
    arr12 = []
    for i in range(len(arr1)):
        arr12.append(arr1[i].tolist())
        arr12[-1].extend(arr2[i].tolist())
    return np.array(arr12, dtype=float)


def snv(input_data):
    """
    Ref. https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/
    """
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
    return output_data


def derivative(X, wl, step_size=3):
    Xdev = []
    hdev = []
    for i in range(X.shape[0]):
        xrow = []
        for j in range(step_size, X.shape[1]-step_size, step_size):
            xrow.append((X[i][j+step_size] - X[i][j-step_size]) / (wl[j+step_size] - wl[j-step_size]))
        Xdev.append(xrow)
    
    for j in range(step_size, X.shape[1]-step_size, step_size):
        hdev.append((wl[j+step_size] + wl[j-step_size])/2.)
    return np.array(Xdev, dtype=float), np.array(hdev, dtype=float)


class FSplit():
    def __init__(self, split_dict):
        self.split_dict = split_dict

    def make_split(self, xdict, x_header, ydict, random_state=0):
        """
        Preprocess descriptors by converting features using SNV scaling
        and then augment features with Savitzky–Golay filter
        
        Ref. https://nirpyresearch.com/savitzky-golay-smoothing-method/
        """
        split = Split()
        split.x_train, split.y_train = make_sample(xdict, ydict, self.split_dict["Cal"])
        split.x_test, split.y_test = make_sample(xdict, ydict, self.split_dict["Tuning"])
        split.x_val, split.y_val = make_sample(xdict, ydict, self.split_dict["Val Ext"])
        
        print("* Calculate SNV ")
        split.x_train = snv(split.x_train)
        split.x_test = snv(split.x_test)
        split.x_val = snv(split.x_val)
        
        print("* Calculate first derivative")
        d1_train, wl_d1 = derivative(split.x_train, np.array(x_header, dtype=float), 1)
        d1_test, _ = derivative(split.x_test, np.array(x_header, dtype=float), 1)
        d1_val, _ = derivative(split.x_val, np.array(x_header, dtype=float), 1)
        
        print("* Calculate second derivative")
        d2_train, wl_d2 = derivative(d1_train, wl_d1, 1)
        d2_test, _ = derivative(d1_test, wl_d1, 1)
        d2_val, _= derivative(d1_val, wl_d1, 1)
        
        print("* Calculate third derivative")
        d3_train, wl_d3 = derivative(d2_train, wl_d2, 1)
        d3_test, _ = derivative(d2_test, wl_d2, 1)
        d3_val, _= derivative(d2_val, wl_d2, 1)
        
        print("* Calculate fourth derivative")
        d4_train, _ = derivative(d3_train, wl_d3, 1)
        d4_test, _ = derivative(d3_test, wl_d3, 1)
        d4_val, _= derivative(d3_val, wl_d3, 1)

        w = 13
        p = 2
        
        print("* Calculate Savitzky–Golay Order 2 first derivative")
        x_train_sgd1 = savgol_filter(split.x_train, w, polyorder = p, deriv=1)
        x_test_sgd1 = savgol_filter(split.x_test, w, polyorder = p, deriv=1)
        x_val_sgd1 = savgol_filter(split.x_val, w, polyorder = p, deriv=1)
        
        print("* Calculate Savitzky–Golay Order 2 second derivative")
        x_train_sgd2 = savgol_filter(split.x_train, w, polyorder = p, deriv=2)
        x_test_sgd2 = savgol_filter(split.x_test, w, polyorder = p, deriv=2)
        x_val_sgd2 = savgol_filter(split.x_val, w, polyorder = p, deriv=2)
        
        # Merge phase
        print("* Merge derivatives")
        split.x_train = mx_join(split.x_train, d1_train)
        split.x_test = mx_join(split.x_test, d1_test)
        split.x_val = mx_join(split.x_val, d1_val)

        split.x_train = mx_join(split.x_train, d2_train)
        split.x_test = mx_join(split.x_test, d2_test)
        split.x_val = mx_join(split.x_val, d2_val)
        
        split.x_train = mx_join(split.x_train, d3_train)
        split.x_test = mx_join(split.x_test, d3_test)
        split.x_val = mx_join(split.x_val, d3_val)
        
        split.x_train = mx_join(split.x_train, d4_train)
        split.x_test = mx_join(split.x_test, d4_test)
        split.x_val = mx_join(split.x_val, d4_val)

        print("* Merge Savitzky–Golay Order 2 first derivative")
        split.x_train = mx_join(split.x_train, x_train_sgd1)
        split.x_test = mx_join(split.x_test, x_test_sgd1)
        split.x_val = mx_join(split.x_val, x_val_sgd1)
        print("* Merge Savitzky–Golay Order 2 second derivative")
        split.x_train = mx_join(split.x_train, x_train_sgd2)
        split.x_test = mx_join(split.x_test, x_test_sgd2)
        split.x_val = mx_join(split.x_val, x_val_sgd2)
        
        print("Training samples: %d x %d features" % (split.x_train.shape[0], split.x_train.shape[1]))
        print("Test samples: %d x %d features" % (split.x_test.shape[0], split.x_test.shape[1]))
        print("Validation samples: %d x %d features" % (split.x_val.shape[0], split.x_val.shape[1]))
        return split


def main():
    if len(sys.argv) != 2:
        print("Usage %s [CSV DM Mango - Features]" % (sys.argv[0]))
    else:
        x_dict, x_header, y_dict, y_header, split_dict = readinput(sys.argv[1])
        fs = FSplit(split_dict)
        res, score_similarity = best_regressor(x_dict, x_header, y_dict, split_fnc=fs.make_split)
        outname = copy(sys.argv[1])
        outname = outname.replace(".csv", "")
        write_results(res, outname)
    return


if __name__ in "__main__":
    main()

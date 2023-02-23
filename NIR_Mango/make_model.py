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


class FSplit():
    def __init__(self, split_dict):
        self.split_dict = split_dict


    def make_split(self, xdict, x_header, ydict, random_state=0):
        split = Split()
        split.x_train, split.y_train = make_sample(xdict, ydict, self.split_dict["Cal"])
        split.x_test, split.y_test = make_sample(xdict, ydict, self.split_dict["Tuning"])
        split.x_val, split.y_val = make_sample(xdict, ydict, self.split_dict["Val Ext"])
        """
        Preprocess descriptors by removing the less important descriptors
        """
        x_all, _ = make_sample(xdict, ydict, None)
        x_all_pre, skip_ids = descriptors_preprocess(x_all, None)
        split.x_train, _ = descriptors_preprocess(split.x_train, skip_ids)
        split.x_test, _ = descriptors_preprocess(split.x_test, skip_ids)
        split.x_val, _ = descriptors_preprocess(split.x_val, skip_ids)
        split.x_header = []
        for i in range(len(x_header)):
            if i in skip_ids:
                continue
            else:
                split.x_header.append(x_header[i])
        print("Training samples: %d x %d features" % (split.x_train.shape[0], split.x_train.shape[1]))
        print("Test samples: %d x %d features" % (split.x_test.shape[0], split.x_test.shape[1]))
        print("Validatiion samples: %d x %d features" % (split.x_val.shape[0], split.x_val.shape[1]))
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

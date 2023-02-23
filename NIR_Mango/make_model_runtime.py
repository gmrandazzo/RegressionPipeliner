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
import time
import csv
import os
from copy import copy
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../" % (dir_path))
from basic_methods import make_split
from basic_methods import pls


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
            if key == "Set":
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


    def make_split(self, xdict, xheader, ydict, random_state=0):
        split = Split()
        split.x_train, split.y_train = make_sample(xdict, ydict, self.split_dict["Cal"])
        split.x_test, split.y_test = make_sample(xdict, ydict, self.split_dict["Tuning"])
        split.x_val, split.y_val = make_sample(xdict, ydict, self.split_dict["Val Ext"])
        return split


def main():
    if len(sys.argv) != 3:
        print("Usage %s [CSV activity_features] [output]" % (sys.argv[0]))
    else:
        x_dict, x_header, y_dict, y_header, split_dict = readinput(sys.argv[1])
        fs = FSplit(split_dict)
        split = fs.make_split(x_dict, x_header, y_dict, 2785)
        st = time.time()
        pls(split)
        et = time.time()
        elapsed_time = et - st
        fo = open(sys.argv[2], "a")
        fo.write('%d,%d,%f\n' % (int(len(x_dict.keys())), int(len(x_header)),  elapsed_time))
        fo.close()
    return


if __name__ in "__main__":
    main()

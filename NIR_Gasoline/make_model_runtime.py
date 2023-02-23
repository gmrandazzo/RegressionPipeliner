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
    x_dict = {}
    y_dict = {}
    x_header = None
    y_header = None
    with open(f_inp) as f:
        r = csv.reader(f, delimiter=',', quotechar='"')
        for row in r:
            key = row[0]
            if key == "Objects":
                y_header = [row[0]]
                x_header = row[1:]
            else:
                y_dict[key] = row[0]
                x_dict[key] = row[1:]
    return x_dict, x_header, y_dict, y_header


def main():
    if len(sys.argv) != 3:
        print("Usage %s [CSV activity_features] [output]" % (sys.argv[0]))
    else:
        x_dict, x_header, y_dict, y_header = readinput(sys.argv[1])
        split = make_split(x_dict, x_header, y_dict, 2785, clean_data=False)
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

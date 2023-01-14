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
import numpy as np
import os
from copy import copy
from pathlib import Path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../" % (dir_path))
from basic_methods import best_regressor
from basic_methods import write_results


def readinput(f_inp):
    x_dict = {}
    x_header = None
    f = open(f_inp, "r", encoding="utf-8")
    for line in f:
        v = str.split(line.strip(), ",")
        if "Molecule" in line:
            x_header = v[1:]
        else:
            x_dict[v[0]] = list()
            for item in v[1:]:
                if len(item) > 0:
                    x_dict[v[0]].append(float(item))
                else:
                    x_dict[v[0]].append(np.nan)
            #x_dict[v[0]] = [float(item) for item in v[1:]]
    f.close()
    if x_header == None:
        x_header = ["Feat%d" % (i+1) for i in range(len(v[1:]))]
    return x_dict, x_header,


def main():
    if len(sys.argv) != 3:
        print("Usage %s [CSV features input] [CSV activities]" % (sys.argv[0]))
    else:
        x_dict, x_header = readinput(sys.argv[1])
        y_dict, y_header = readinput(sys.argv[2])
        for j in range(len(y_header)):
            kin = {}
            for key in y_dict.keys():
                if np.isnan(y_dict[key][j]) == False:
                    kin[key] = -1*np.log10(y_dict[key][j]+1)
                else:
                    continue
            if len(kin.keys()) >= 40:
                if Path("%s.json" % (y_header[j])).is_file() == True:
                    print("%s calculated" % (y_header[j]))
                else:
                    res, _ = best_regressor(x_dict, x_header, kin)
                    write_results(res, y_header[j])
                    os.remove("emissions.csv")
            else:
                continue
    return

if __name__ in "__main__":
    main()

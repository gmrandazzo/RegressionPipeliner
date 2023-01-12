#!/usr/bin/env python3

import json
import sys

def main():
    f = open(sys.argv[1], "r")
    data = json.load(f)
    pre_name = sys.argv[1].replace(".json", "")
    for key in data.keys():
        print("%s_%s,%f,%f,%f,%e" % (pre_name, key, data[key]["MSE"], data[key]["MAE"], data[key]["R2"], data[key]["emission"]))
    f.close()

if __name__ in "__main__":
    main()

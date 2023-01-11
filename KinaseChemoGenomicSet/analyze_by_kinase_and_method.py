#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt

def readinp(finp):
    d = {}
    f = open(finp, "r")
    for line in f:
        v = str.split(line.strip(), ",")
        d[v[0]] = v[1:]
    f.close()
    return d

def process(finp):
    d = readinp(finp)
    methods=["Linear-SVR", "RBF-SVR", "DecisionTree", "ExtraTrees", "RandomForest", "KNeighbors", "PLS", "PLS-SKLEARN", "XGBoost", "CatBoost", "DNN"]
    res = {}
    for m in methods:
        mse = []
        mae = []
        rsq=[]
        emis=[]
        kname = []
        for key in d.keys():
            if m in key:
                mse.append(d[key][0])
                mae.append(d[key][1])
                rsq.append(d[key][2])
                emis.append(d[key][3])
                name = key.split("/")[1].replace(m, "")
                kname.append(name)
            else:
                continue
        mse = np.array(mse).astype(float)
        mae = np.array(mae).astype(float)
        rsq = np.array(rsq).astype(float)
        emis = np.array(emis).astype(float)
        res[m] = {"MSE": mse, "MAE": mae, "R2": rsq, "Emissions": emis, "names": kname}
    return res

def plot(res, outfig):
    for m in res.keys():
        fig, ax = plt.subplots()
        ids = range(len(res[m]["names"]))
        bars = ax.barh(ids, res[m]["R2"])
        labels = []
        for i, v in enumerate(res[m]["R2"]):
            if v > 0.65:
                labels.append(res[m]["names"][i])
            else:
                labels.append("")
        ax.bar_label(bars, labels=labels, padding=8, color='b', fontsize=6)
        ax.set_ylabel("Kinase")
        ax.set_xlabel("R2")
        ax.set_title("%s" % (m))
        ax.set_xlim([0,1])
        plt.savefig("%s-%s" % (m.replace(" ", "_"), outfig), dpi=300)


def main():
    res1=process(sys.argv[1])
    plot(res1, sys.argv[2])

if __name__ in "__main__":
    main()

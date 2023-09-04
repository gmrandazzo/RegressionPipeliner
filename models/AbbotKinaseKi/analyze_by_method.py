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
        for key in d.keys():
            if m in key:
                mse.append(d[key][0])
                mae.append(d[key][1])
                rsq.append(d[key][2])
                emis.append(d[key][3])
            else:
                continue
        mse = np.array(mse).astype(float)
        mae = np.array(mae).astype(float)
        rsq = np.array(rsq).astype(float)
        emis = np.array(emis).astype(float)
        res[m] = {"MSE": mse.mean(), "MAE": mae.mean(), "R2": rsq.mean(), "Emissions": emis.mean()}
    return res

def plot(res, outfig, title):
    x = []
    y = []
    methods = list(res.keys())
    np.random.seed(1234)
    c = np.random.rand(len(methods))
    area = []
    for key in methods:
        x.append(res[key]["MSE"])
        y.append(res[key]["R2"])
        area.append((5e7*res[key]["Emissions"]))
    plt.scatter(x, y, s=area, c=c, alpha=0.5)
    plt.title(title)
    plt.xlabel("MSE")
    plt.ylabel("R2")
    size = 6
    for i, txt in enumerate(methods):
       plt.annotate(txt, (x[i], y[i]), fontsize=size)
    plt.savefig(outfig, dpi=300)


def main():
    res1=process(sys.argv[1])
    plot(res1, sys.argv[2], sys.argv[3])

if __name__ in "__main__":
    main()

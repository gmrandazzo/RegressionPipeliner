import json
import hashlib
import numpy as np
from regressionpipeliner import dataset

def test_make_split():
    np.random.seed(872674)
    a = np.random.rand(100,21)
    xdata = {}
    xheader = [f"col{j}" for j in range(20)]
    ydata = {}
    yheader = []
    for i in range(100):
        xdata[f'obj{i}'] = a[i][:-1]
        ydata[f'obj{i}'] = a[i][-1]
    split = dataset.make_split(xdata, xheader, ydata, 872674)
    res = {"x_train": "42b1a216e44284aa77cd997e7c0d424d6c713414ff9656a9c4f795c9019cbf3429e934a7db3ba45d934b15d79eb85c5a2c73fd519c46f38dbc698a782c1cfce9",
           "y_train":"d181e56688cd8118eb5f12cd1e86e29c7a1d46bbb0249c8901700366ac8fc443ef9daf29959fa3e549e9c3ac7154b7d8ab49e68d30f6f8ffc9626baa4e66c27f",
           "x_test":"1e13a01103b8eff07037bea8e627572b6fc4192f81fa901311cd59a60ea309b9c4269bceea51c40387e317f9e5ef934aa904740827aafde930bb21ddb2027eb9",
           "y_test":"71c699024c7a79ed102931d0dbaa2880e58005fffbf6094221cf599b4b4df126fbe0aa2ec1ae6ce75db057af5b857dc9b8cba75506822f977bff80aa3a9112a0",
           "x_val":"e6b8c12851c49bf06e662e16d5dc2ade8e233e61b33bbf32d6e48563aebcbe60f0bcd0b7af85df5bf454e8d304e2ce859afd1dc9d798620366b92d10425148e8",
           "y_val":"f1e110c9b8b871533a79d20de64e313aec01232f0fef35237624c4355da41349b66c17db1afd0f5f4f243c67515a8d4770f014afa4c0326582e6ba818613d74b"}
    assert hashlib.sha512(split.x_train.tobytes()).hexdigest() == res["x_train"]
    assert hashlib.sha512(split.y_train.tobytes()).hexdigest() == res["y_train"]
    assert hashlib.sha512(split.x_test.tobytes()).hexdigest() == res["x_test"]
    assert hashlib.sha512(split.y_test.tobytes()).hexdigest() == res["y_test"]
    assert hashlib.sha512(split.x_val.tobytes()).hexdigest() == res["x_val"]
    assert hashlib.sha512(split.y_val.tobytes()).hexdigest() == res["y_val"]
    
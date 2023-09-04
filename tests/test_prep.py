import hashlib
from regressionpipeliner import prep

def test_get_low_variance_descriptors():
    mx = [[1223, 1.],[0, 1.], [1, 1.], [-1223, 1.0]]
    assert  prep.get_low_variance_descriptors(mx)[0] == 1

def test_descriptors_preprocess():
    mx = [[1223, 1.],[0, 1.], [1, 1.], [-1223, 1.0]]
    mx_prep, skip_list = prep.descriptors_preprocess(mx)
    assert hashlib.sha512(mx_prep).hexdigest() == "a805079c872cce78e2795f50d908d803464a9d8c4242858170eb8f95761470c825417aa5a22d05fb6992cb7dd60481f078819444bf53b56ce386ee707d00f91a"

#!/usr/bin/pypy

from math import exp, log, sqrt
from pymmh3 import hash

#signed = False    # Use signed hash? Set to False for to reduce number of hash calls

# A. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-17), 10e-17)        # The bounds
    return -log(p) if y == 1. else -log(1. - p)

# B. Apply hash trick of the original csv row
# for simplicity, we treat both integer and categorical features as categorical
# INPUT:
#     csv_row: a csv dictionary, ex: {'Lable': '1', 'I1': '357', 'I2': '', ...}
#     D: the max index that we can hash to
# OUTPUT:
#     x: a list of indices that its value is 1
def get_x(row, D):
    x = {}
    x['sane'] = True

    try:
        indices = map(lambda _x: hash(_x) % (D - 1), row)
    except:
        x['sane'] = False
        return x

    x[D - 1] = 1  # D - 1 is the index of the bias term
    for i in indices:
        if(not x.has_key(i)):
            x[i] = 0
#        if signed:
#        try:
#            x[index] += (1 if (hash(s + '@')%2)==1 else -1) # Disable for speed
#       except:
#                print 'ERROR IN :(',key, "=", value,') AT',csv_row
#            return -1
#        else:
        x[i] += 1
    return x  # x contains indices of features that have a value as number of occurences


# C. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def get_p(x, w):
    if not x['sane']:
        return [.5, 0.]
    del x['sane']

    wTx = 0.
    for i, xi in x.items():
        wTx += w[i] * xi  # w[i] * x[i]

    x['sane'] = True
    return [1./(1. + exp(-max(min(wTx, 50.), -50.))), wTx]  # bounded sigmoid


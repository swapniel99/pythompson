#!/usr/bin/pypy

from math import exp, log, sqrt
from pymmh3 import hash
from numpy.random import normal, binomial, choice

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

    try:
        indices = map(lambda _x: hash(_x) % (D - 1), row)
    except:
        x['insane'] = True
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

def add2x(x, c, D):
    try:
        indices = map(lambda _x: hash(_x) % (D - 1), c)
    except:
        x['insane'] = True
        return x

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
    if x.has_key('insane'):
        return [.5, 0.]

    wTx = 0.
    for i, xi in x.items():
        wTx += w[i] * xi  # w[i] * x[i]

    return [1./(1. + exp(-max(min(wTx, 100.), -100.))), wTx]  # bounded sigmoid

# D. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def get_p_ts(x, m, sd):
    if x.has_key('insane'):
        return [.5, 0.]

    wTx = 0.
    for i, xi in x.items():
        wi = normal(m[i], sd[i])
        wTx += wi * xi  # w[i] * x[i]

    return [1./(1. + exp(-max(min(wTx, 100.), -100.))), wTx]  # bounded sigmoid

# E. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def get_p_linucb(x, m, q, alpha):
    if x.has_key('insane'):
        return [.5, 0.]

    mTx = 0.
    qx = 0.
    for i, xi in x.items():
        qx += xi * xi / q[i]
        mTx += mi * xi  # w[i] * x[i]

    score = mTx + alpha * sqrt(qx)
    return [1./(1. + exp(-max(min(score, 100.), -100.))), score]  # bounded sigmoid

def getrev(campattr, y):
    cvr = .05 # Assumption
    if y != 1 and y != 0:
        print "EXCEPTION! y =", y
    camptype = campattr[2]
    if camptype == 1:
        return campattr[3]
    elif camptype == 2:
        return ((1000. * campattr[4]) if y == 1 else 0.)
    elif camptype == 3:
        return ((1000. * campattr[5] * cvr) if y == 1 else 0.)
    else:
        return 0.

def ts_selectcamp(req, qc, qcl, m, sd, D, campdet): # Selects based on max CTR
    x = get_x(req, D)
    cases = map(lambda c: add2x(x, c, D), qcl)
    indices = [index for case in cases for index in case]
    w = {}
    for i in indices:
        if not w.has_key(i):
            w[i] = normal(m[i], sd[i])
    preds = map(lambda c: get_p(c, w)[0], cases)
    attrs = map(lambda c: campdet[c], qc)
    ecpms = map(lambda (a, p): calcecpm(p, a), zip(attrs, preds))
    cp = [[c, p, a, e] for (c, p, a, e) in zip(cases, preds, attrs, ecpms) if not c.has_key('insane')]
    return max(cp, key = lambda x: x[1])

def calcecpm(ctr, campattr):
    cvr = .05 # Assumption
    camptype = campattr[2]
    if camptype == 1:
        return campattr[3]
    elif camptype == 2:
        return (campattr[4] * ctr * 1000.)
    elif camptype == 3:
        return (campattr[5] * ctr * cvr * 1000.)
    else:
        return 0.

def softmax_selectcamp_ecpm(req, qc, qcl, m, D, campdet):
    tau = .2
    x = get_x(req, D)
    cases = map(lambda c: add2x(x, c, D), qcl)
    preds = map(lambda c: get_p(c, m)[0], cases)
    attrs = map(lambda c: campdet[c], qc)
    ecpms = map(lambda (a, p): calcecpm(p, a), zip(attrs, preds))
    weights = map(lambda e: exp(-max(min(e / tau, 700.), -700.)), ecpms)
    cp = [[c, p, a, e, w] for (c, p, a, e, w) in zip(cases, preds, attrs, ecpms, weights) if not c.has_key('insane')]
    sumwts = sum([c[4] for c in cp])
    probs = map(lambda x: x[4] / sumwts, cp)
    selind = choice(len(probs), p = probs)
    return cp[selind][0:4]

def ts_selectcamp_ecpm(req, qc, qcl, m, sd, D, campdet):
    x = get_x(req, D)
    cases = map(lambda c: add2x(x, c, D), qcl)
    indices = [index for case in cases for index in case]
    w = {}
    for i in indices:
        if not w.has_key(i):
            w[i] = normal(m[i], sd[i])
    preds = map(lambda c: get_p(c, w)[0], cases)
    attrs = map(lambda c: campdet[c], qc)
    ecpms = map(lambda (a, p): calcecpm(p, a), zip(attrs, preds))
    cp = [[c, p, a, e] for (c, p, a, e) in zip(cases, preds, attrs, ecpms) if not c.has_key('insane')]
    return max(cp, key = lambda x: x[3])

def getclick(x, Wreal):
    if x.has_key('insane'):
        print "WHAT WHAT WHAT?!!!"
        return 0
    preal = get_p(x, Wreal)[0]
    return binomial(1, preal)


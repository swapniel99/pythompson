#!/usr/bin/pypy

import sys
from datetime import datetime
from itertools import islice
from math import sqrt
from functions import logloss, get_p, ts_selectcamp, getclick, ts_selectcamp_ecpm, softmax_selectcamp_ecpm, getrev
import pickle
import gc

# parameters #################################################################

train = sys.stdin
inmodel = sys.argv[1]
outmodel = sys.argv[2]

fresh = False # SET inmodel IF fresh == False

alpha = .1 # .03  # Initial learning rate. Need to explore for live.. # .1
passes = 2
adapt = 1

#tsalpha = .5

batchsize = 1000000
logbatch = batchsize/10

# Get campaignid to clientid mappings
f = open('campaigndetails.dict','r')
campdet = pickle.load(f)
f.close()

if fresh:    
    lambda2 = .0001   # L2 regularization
    D = 2 ** 24    # number of weights use for learning
    m = [0.] * D
    q = [lambda2] * D
    header = ['z','d','hr','wd','zr','sa','as','con']
    extra = ['camp','cln']
    f = open('wreal24','r')
    Wreal = pickle.load(f)
    f.close()
else:
    f = open(inmodel,'r')
    pars = pickle.load(f)
    f.close()
    D = pars['D']
    m = pars['m']
    q = pars['q']
    header = pars['header']
    extra = pars['extra']
    Wreal = pars['Wreal']

# D. Update given model
# INPUT:
#     w: weights
#     n: a counter that counts the number of times we encounter a feature
#        this is used for adaptive learning rate
#     x: feature
#     p: prediction of our model
#     y: answer
# OUTPUT:
#     w: updated model
#     n: updated count
def update_w(w, g, x, p, y, m, q):
    for i, xi in x.items():
        # alpha / (sqrt(g) + 1) is the adaptive learning rate heuristic
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1
        delta = (p - y) * xi + (w[i] - m[i]) * .0001 # q[i]
        if adapt > 0:
            g[i] += delta ** 2
        w[i] -= delta * alpha / (sqrt(g[i]) ** adapt)  # Minimising log loss
    return w, g

# D. Update given model
# INPUT:
#     w: weights
#     n: a counter that counts the number of times we encounter a feature
#        this is used for adaptive learning rate
#     x: feature
#     p: prediction of our model
#     y: answer
# OUTPUT:
#     w: updated model
#     n: updated count
def update_m(X, y, m, q, D, trpasses = 1):
    w = list(m)
    g = [0.] * D
    t1 = 0
    loss1 = 0.
    lossb1 = 0.
    for i in range(trpasses):
        for j in range(len(X)):
            if X[j].has_key('insane'):
                continue
            p = get_p(X[j], w)[0]
#            lossx1 = logloss(p, y[j])
#            loss1 += lossx1
#            lossb1 += lossx1
#            t1 += 1
#            if t1 % logbatch == 0 and t1 > 1:
#                print('%s\tPass: %d\tTraining encountered: %d\tcurrent whole logloss: %f\tcurrent batch logloss: %f' % (datetime.now(), i, t1, loss1/t1, lossb1/logbatch))
#                lossb1 = 0.
            w, g = update_w(w, g, X[j], p, y[j], m, q)
    del g
    return w

def update_q(X, w, q):
    ps = map(lambda x: get_p(x, w)[0], X)
    for (p_, x_) in zip(ps, X):
        if x_.has_key('insane'):
            continue
        for i, xi in x_.items():
            q[i] += p_ * (1 - p_) * xi
    del ps
    return q


# training #######################################################

# start training a logistic regression model using on pass sgd
loss = 0.
lossb = 0.
erev = 0.
erevb = 0.
rev = 0.
revb = 0.
t = 0

while True:
    lines = list(islice(train, batchsize))
    
    if not lines:
        break
    
    lenbatch = len(lines)

    rows = map(lambda x: x.strip().split('^'), lines)
    del lines

    reqs = map(lambda x: [a + '=' + b for (a, b) in zip(header, x[1:9])], rows)
    qualcamps = map(lambda x: filter(lambda x_: x_ != '', x[11].strip('~,').split(',')), rows)
    qualcamplvls = map(lambda x: map(lambda x_: [extra[0] + '=' + x_, extra[1] + '=' + (campdet[x_][1] if campdet.has_key(x_) else '')], x), qualcamps)
    del rows

    sd = 0
#    sd = map(lambda x: tsalpha / sqrt(x), q)

#    Xp = [ts_selectcamp(req, qc, qcl, m, sd, D, campdet) for (req, qc, qcl) in zip(reqs, qualcamps, qualcamplvls) if not len(qcl) == 0]
#    Xp = [ts_selectcamp_ecpm(req, qc, qcl, m, sd, D, campdet) for (req, qc, qcl) in zip(reqs, qualcamps, qualcamplvls) if not len(qc) == 0]
    Xp = [softmax_selectcamp_ecpm(req, qc, qcl, m, D, campdet) for (req, qc, qcl) in zip(reqs, qualcamps, qualcamplvls) if not len(qc) == 0]
    del sd, reqs, qualcamps

    X = map(lambda x: x[0], Xp)
    erevb = sum(map(lambda x: x[3], Xp))

    y = map(lambda x: getclick(x, Wreal), X)
    revb = sum(map(lambda (x, y_): getrev(x[2], y_), zip(Xp, y)))

    lossb = sum([logloss(xp_[1], y_) for (xp_, y_) in zip(Xp, y)])
    del Xp

    loss += lossb
    erev += erevb
    rev += revb

    t += lenbatch
    print('%s\tEncountered: %d\tLogloss: %f\tBatch Logloss: %f\tECPM: %f\t Batch ECPM: %f\tRev: %f\tBatch Rev: %f' % (datetime.now(), t, loss/t, lossb/lenbatch, erev/t, erevb/lenbatch, rev/t, revb/lenbatch))

    w = update_m(X, y, m, q, D, passes)
    del m, y
    m = w
#    q = update_q(X, w, q)
    del X
    gc.collect()

print "Final logloss:",loss/t

pars2 = {}
pars2['D'] = D
pars2['header'] = header
pars2['m'] = w
pars2['q'] = q
pars2['Wreal'] = Wreal

print "Saving model..."

dumpfile=open(outmodel,'w')
pickle.dump(pars2,dumpfile)
dumpfile.close()


#!/usr/bin/pypy

import sys
from datetime import datetime
from itertools import islice
from math import sqrt
from functions import logloss, get_x, get_p
import pickle

# parameters #################################################################

train = sys.stdin
outmodel = sys.argv[1]
inmodel = ''

alpha = .1  #0.09 # Initial learning rate.

fresh = True # SET inmodel IF fresh == False

adapt = 1

batchsize = 1
logbsize = 1000000

if fresh:    
    lambda2 = .0002   # L2 regularization
    D = 2 ** 24    # number of weights use for learning
    m = [0.] * D
    q = [lambda2] * D
    header = ['z','d','hr','wd','zr','sa','as','con','camp','cln']
else:
    f = open(inmodel,'r')
    pars = pickle.load(f)
    f.close()
    D = pars['D']
    m = pars['m']
    q = pars['q']
    header = pars['header']

# initialize new model
w = list(m)
g = [0.] * D
#q[D - 1] = 0. # No regularisation for bias 

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
def update_w(w, g, X, p, y, m, q):
    upd = {}
    for j in range(len(X)):
        if X[j].has_key('insane'):
            continue
        for i, xi in X[j].items():
            # alpha / (sqrt(g) + 1) is the adaptive learning rate heuristic
            # (p - y) * x[i] is the current gradient
            # note that in our case, if i in x then x[i] = 1
            delta = (p[j] - y[j]) * xi + (w[i] - m[i]) * q[i]
            if adapt > 0:
                g[i] += delta ** 2
            if not upd.has_key(i):
                upd[i] = 0.
            upd[i] += delta #/ batchsize # Minimising log loss
    for i, ui in upd.items():
        w[i] -= upd[i] * alpha / (sqrt(g[i]) ** adapt)
    return w, g

# training #######################################################

# start training a logistic regression model using on pass sgd
loss = 0.
lossb = 0.
errcount = 0
t = 0

while True:
    lines = list(islice(train, batchsize))
    
    if not lines:
        break
    
    lenbatch = len(lines)

    rows = map(lambda x: x.strip().split('^'), lines)

    y = map(lambda x: 1. if x[0] == '1' else 0., rows)
    X = map(lambda x: get_x([a + '=' + b for (a, b) in zip(header, x[1:11])], D), rows)

    errorbatch = sum(map(lambda x: 1 if x.has_key('insane') else 0, X))
    errcount += errorbatch

    p = map(lambda x: get_p(x, w)[0], X)

    closs = sum([logloss(p_, y_) for (p_, y_) in zip(p, y)])
    lossb += closs
    loss += closs

    t += lenbatch
    if t % logbsize == 0 and t > 1:
        print('%s\tEncountered: %d\tTotal logloss: %f\tCurrent logloss: %f' % (datetime.now(), t, loss/t, lossb/logbsize))
        lossb = 0.
#    print "Encountered:",t,"Current loss:",lossb/lenbatch,"Total loss:",loss/t
    w, g = update_w(w, g, X, p, y, m, q)

print "Final logloss:",loss/t
print "ERROR ROWS:",errcount

## UPDATE q

pars2 = {}
pars2['D'] = D
pars2['header'] = header
pars2['m'] = w
pars2['q'] = q

print "Saving model..."

dumpfile=open(outmodel,'w')
pickle.dump(pars2,dumpfile)
dumpfile.close()


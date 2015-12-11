#!/usr/bin/pypy

import sys
from datetime import datetime
from csv import DictReader
from math import sqrt
from functions import logloss, get_x, get_p
import pickle
import numpy as np

# parameters #################################################################

train = sys.stdin
outmodel = sys.argv[1]
inmodel = ''

alpha = .1  #0.09 # Initial learning rate.
logbatch = 1000000

fresh = True # SET inmodel IF fresh == False

if fresh:    
    lambda2 = .0002   # L2 regularization
    D = 2 ** 18    # number of weights use for learning
    m = np.zeros(D)
    q = np.repeat(lambda2, D)
    header = np.arr(['clk','z','d','hr','wd','zr','sa','as','con','camp','cln','camplist','algo'])
    extra = np.arr(['camplist','algo'])
else:
    f = open(inmodel,'r')
    pars = pickle.load(f)
    f.close()
    D = pars['D']
    m = pars['m']
    q = pars['q']
    header = pars['header']
    extra = pars['extra']

# initialize new model
w = np.copy(m)
g = np.zeros(D)
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
def update_w(w, g, x, p, y, m, q):
    for i, xi in x.items():
        # alpha / (sqrt(g) + 1) is the adaptive learning rate heuristic
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1
        delta = (p - y) * xi + (w[i] - m[i]) * q[i]
        g[i] += delta ** 2
        w[i] -= delta * alpha / sqrt(g[i])  # Minimising log loss
    return w, g

# training #######################################################

# start training a logistic regression model using on pass sgd
loss = 0.
lossb = 0.
errcount = 0
#for t, row in enumerate(DictReader(train, header, delimiter='^')):
while True:
    lines = list(islice(train, 100))
    if not lines:
        break
    rows = np.genfromtxt(lines, dtype=str, delimiter='^')



    y = 1. if row['clk'] == '1' else 0.

    del row['clk']  # can't let the model peek the answer

    for h in (extra):
        del row[h]

    # main training procedure
    # step 1, get the hashed features
    x = get_x(row, D)
    if x == -1:
        errcount += 1
        continue
    # step 2, get prediction
    p, _ = get_p(x, w)

    # for progress validation, useless for learning our model
    lossx = logloss(p, y)
    loss += lossx
    lossb += lossx
    if t % logbatch == 0 and t > 1:
        print('%s\tencountered: %d\tcurrent whole logloss: %f\tcurrent batch logloss: %f' % (datetime.now(), t, loss/t, lossb/logbatch))
        lossb = 0.

    # step 3, update model with answer
    w, g = update_w(w, g, x, p, y, m, q)

print "Final logloss:",loss/t
print "ERROR ROWS:",errcount

## UPDATE q

pars = {}
pars['D'] = D
pars['header'] = header
pars['extra'] = extra
pars['m'] = w
pars['q'] = q

print "Saving model..."

dumpfile=open(outmodel,'w')
pickle.dump(pars,dumpfile)
dumpfile.close()


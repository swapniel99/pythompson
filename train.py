#!/usr/bin/pypy

import sys, json
from datetime import datetime
from math import sqrt
from functions import logloss, get_x, get_p, readvw, compress, decompress

# parameters #################################################################

train = sys.stdin
outmodel = sys.argv[1]
inmodel = ''

alpha = .1  #0.09 # Initial learning rate. 10x rate if taking 10% sample of training data.

fresh = True # SET inmodel IF fresh == False

adapt = 1
fudge = 0.

logbatch = 10000000

if fresh:
    lambda2 = .0002   # L2 regularization
    D = 2 ** 20    # number of weights use for learning
    w = [0.] * D #m = [0.] * D
    meta = {}
#    q = [lambda2] * D
#    header = ['z','d','hr','wd','zr','sa','as','con','camp','cln']
else:
    f = open(inmodel,'r')
    pars = json.load(f)
    f.close()
    D = pars['D']
    w = decompress(pars['w'], D) #m = pars['m']
    meta = pars['meta']
#    q = pars['q']
#    header = pars['header']

# initialize new model
#w = list(m)
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
def update_w(w, g, x, p, y):
    for i, xi in x.items():
        # alpha / (sqrt(g) + 1) is the adaptive learning rate heuristic
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1
#        delta = (p - y) * xi + (w[i] - m[i]) * lambda2 #q[i]
        delta = (p - y) * xi + w[i] * lambda2 #q[i]
        if adapt > 0:
            g[i] += delta ** 2
        w[i] -= delta * alpha / ((fudge + sqrt(g[i])) ** adapt)  # Minimising log loss
#        counts[i] += 1
    return w, g

# training #######################################################
# start training a logistic regression model using on pass sgd
loss = 0.
lossb = 0.
errcount = 0
t = 0

for line in train:
    # READS VW FORMAT
    row = readvw(line)
    
    y = 1. if row[0] == '1' else 0.
    x = get_x(row[1:12], D)

    if x.has_key('insane'):
        errorcount += 1
        continue

    p = get_p(x, w)[0]

    # for progress validation, useless for learning our model
    lossx = logloss(p, y)
    loss += lossx
    lossb += lossx
    t += 1
    if t % logbatch == 0 and t > 1:
        print('%s\tencountered: %d\tcurrent whole logloss: %f\tcurrent batch logloss: %f' % (datetime.now(), t, loss/t, lossb/logbatch))
        sys.stdout.flush()
        lossb = 0.

    # step 3, update model with answer
#    w, g = update_w(w, g, x, p, y, m, q)
    w, g = update_w(w, g, x, p, y)

print "Final logloss:",loss/t
print "ERROR ROWS:",errcount

pars2 = {}
pars2['D'] = D
#pars2['header'] = header
pars2['w'] = compress(w, D)
#pars2['q'] = q

print "Saving model..."

dumpfile=open(outmodel, 'w')
json.dump(pars2, dumpfile, indent=1, separators=(',', ':'))
dumpfile.close()


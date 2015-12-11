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
logbatch = 1000000

fresh = True # SET inmodel IF fresh == False

if fresh:    
    lambda2 = .0002   # L2 regularization
    D = 2 ** 8    # number of weights use for learning
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
lossbb = 0.
errcount = 0
#for t, row in enumerate(DictReader(train, header, delimiter='^')):
while True:
    lines = list(islice(train, 2))
    
    if not lines:
        break
    
    lenbatch = len(lines)

    rows = map(lambda x: x.strip().split('^'), lines)

    y = map(lambda x: 1. if x[0] == '1' else 0., rows)
    reqs = map(lambda x: [a + '=' + b for (a, b) in zip(header, x[1:11])], rows)

    X = map(lambda x: get_x(x, D), reqs)

    errorbatch = sum(map(lambda x: 0 if x['sane'] else 1, X))
    errcount += errorbatch

    p = map(lambda x: get_p(x, w)[0], X)

    lossb = sum([logloss(p_, y_) for (p_, y_) in zip(p, y)])
    loss += lossb

    w, g = update_w(w, g, X, p, y, m, q)
# UPDATE FUNCTION NEEDS TO BE VECTORIZED
'''
    if t % logbatch == 0 and t > 1:
        print('%s\tencountered: %d\tcurrent whole logloss: %f\tcurrent batch logloss: %f' % (datetime.now(), t, loss/t, lossb/logbatch))
        lossb = 0.

    # step 3, update model with answer
    w, g = update_w(w, g, x, p, y, m, q)

print "Final logloss:",loss/t
print "ERROR ROWS:",errcount
'''
## UPDATE q

pars = {}
pars['D'] = D
pars['header'] = header
pars['m'] = w
pars['q'] = q

print "Saving model..."

dumpfile=open(outmodel,'w')
pickle.dump(pars,dumpfile)
dumpfile.close()


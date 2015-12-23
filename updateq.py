#!/usr/bin/pypy

import sys
from datetime import datetime
from itertools import islice
from functions import logloss, get_x, get_p

test = sys.stdin
inmodel = sys.argv[1]
outmodel = sys.argv[2]

batchsize = 1000000

import pickle
f = open(inmodel,'r')
pars = pickle.load(f)
f.close()

D = pars['D']
w = pars['m']
q = pars['q']
header = pars['header']
Wreal = pars['Wreal']
extra = ['camp' + 'cln']

# testing (build preds file)
loss = 0.
lossb = 0.
errcount = 0
t = 0

while True:
    lines = list(islice(test, batchsize))
    if not lines:
        break

    X = filter(lambda x1: not x1.has_key('insane'), map(lambda x: get_x([(a + '=' + b) for (a, b) in zip(header + extra, x.strip().split('^')[1:11])], D), lines))
   
    p = map(lambda x: get_p(x, w)[0], X)

    for (p_, x_) in zip(p, X):
        for i, xi in x_.items():
            q[i] += p_ * (1. - p_) * xi

    t += len(lines)
    print str(datetime.now()) + " Encountered:",t

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


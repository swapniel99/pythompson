#!/usr/bin/pypy

import sys
from math import sqrt
from numpy.random import normal

inmodel = sys.argv[1]
outmodel = sys.argv[2]

import pickle
f = open(inmodel,'r')
pars = pickle.load(f)
f.close()

D = pars['D']
m = pars['m']
q = pars['q']
header = pars['header']
Wreal = [0.] * D
extra = ['camp', 'cln']

# testing (build preds file)
for i in range(D):
    Wreal[i] = normal(m[i], 1. / sqrt(q[i]))

pars2 = {}
pars2['D'] = D
pars2['header'] = header
pars2['m'] = m
pars2['q'] = q
pars2['Wreal'] = Wreal
pars2['extra'] = extra

print "Saving model..."

dumpfile=open(outmodel,'w')
pickle.dump(pars2,dumpfile)
dumpfile.close()


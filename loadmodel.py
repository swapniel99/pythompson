#!/usr/bin/pypy

import sys
import numpy as np
from math import sqrt

modelfile = sys.argv[1]

import pickle
f = open(modelfile,'r')
pars = pickle.load(f)
f.close()

w = pars['m']
q = pars['q']
D = pars['D']
counts = pars['counts']

for i in range(D):
    if not w[i] == 0.:
        print i, w[i], q[i], sqrt(1./q[i]), counts[i]


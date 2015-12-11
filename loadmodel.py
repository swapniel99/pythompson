#!/usr/bin/pypy

import sys
import numpy as np

modelfile = sys.argv[1]

import pickle
f = open(modelfile,'r')
pars = pickle.load(f)
f.close()

w = pars['w']

x = filter(lambda x: x != 0., w)

#print np.mean(w), np.var(w)
print np.var(x)



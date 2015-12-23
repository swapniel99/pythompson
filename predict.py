#!/usr/bin/pypy

import sys
from datetime import datetime
from math import sqrt
from functions import logloss, get_x, get_p_ts, get_p_linucb
import pickle

# parameters #################################################################

stream = sys.stdin
inmodel = sys.argv[1]
outfile = open(sys.argv[2], 'w')

logbatch = 1000000

tsalpha = .5      # Exploration control. Paper suggested in range {0.25, 0.5}
linucbalpha = 2. # 2 was used in paper.

f = open(inmodel,'r')
pars = pickle.load(f)
f.close()
D = pars['D']
m = pars['m']
q = pars['q']
header = pars['header']

sd = map(lambda x: tsalpha / sqrt(x), q)

# training #######################################################
# start training a logistic regression model using on pass sgd
loss = 0.
lossb = 0.
errcount = 0
t = 0

for line in stream:
    row = line.strip().split('^')
    
    y = 1. if row[0] == '1' else 0.
    x = get_x([a + '=' + b for (a, b) in zip(header, row[1:11])], D)

    if x.has_key('insane'):
        errorcount += 1
        continue

    p = get_p_ts(x, m, sd)[0]
#    p = get_p_linucb(x, m, q, linucbalpha)[0]

    # for progress validation, useless for model
    lossx = logloss(p, y)
    loss += lossx
    lossb += lossx
    t += 1
    if t % logbatch == 0 and t > 1:
        print('%s\tencountered: %d\tcurrent whole logloss: %f\tcurrent batch logloss: %f' % (datetime.now(), t, loss/t, lossb/logbatch))
        lossb = 0.
    outfile.write('%d %f\n' % (int(y), p))

outfile.close()

print "Final logloss:",loss/t
print "ERROR ROWS:",errcount

with open('tsscores5','a') as f:
    f.write("test: " + str(loss/t) + "\n")


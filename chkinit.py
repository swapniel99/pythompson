#!/usr/bin/pypy

import sys
from datetime import datetime
from itertools import islice
from functions import logloss, get_x, get_p, getclick

test = sys.stdin
modelfile = sys.argv[1]
#outfile = open(sys.argv[2], 'w')

batchsize = 1000000

comment = sys.argv[2]

import pickle
f = open(modelfile,'r')
pars = pickle.load(f)
f.close()

D = pars['D']
w = pars['m']
header = pars['header']
extra = ['camp','cln'] # pars['extra']
Wreal = pars['Wreal']

# testing (build preds file)
loss = 0.
lossb = 0.
errcount = 0
t = 0.
while True:
    lines = list(islice(test, batchsize))
    if not lines:
        break

    lenbatch = len(lines)

    rows = map(lambda x: x.strip().split('^'), lines)

    X = map(lambda x: get_x([a + '=' + b for (a, b) in zip(header + extra, x[1:11])], D), rows)

    y = map(lambda x: getclick(x, Wreal), X)

    errorbatch = sum(map(lambda x: 1 if x.has_key('insane') else 0, X))
    errcount += errorbatch

    p = map(lambda x: get_p(x, w)[0], X)
#    p = map(lambda x: x[0], pr)
#    r = map(lambda x: x[1], pr)

    lossb = sum([logloss(p_, y_) for (p_, y_) in zip(p, y)])
    loss += lossb

    t += lenbatch
    print str(datetime.now()) + " Encountered:",t,"Current loss:",lossb/lenbatch,"Total loss:",loss/t

#    for i in range(lenbatch):
#        outfile.write('%d %f\n' % (int(y[i]), r[i]))

#outfile.close()

print 'Error count:',errcount
print ('FINAL AVG LOSS = %f' % (loss/t))

with open('initscores','a') as f:
    f.write("lambda "+ comment+ ": " + str(loss/t) + "\n")


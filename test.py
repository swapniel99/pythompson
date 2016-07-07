#!/usr/bin/pypy

import sys, json
from datetime import datetime
from itertools import islice
from functions import logloss, get_x, get_p, readvw, decompress

# parameters #################################################################

test = sys.stdin
modelfile = sys.argv[1]
outfile = open(sys.argv[2], 'w')

batchsize = 500000

#comment = sys.argv[3]

f = open(modelfile,'r')
pars = json.load(f)
f.close()

D = pars['D']
initial_wt = pars['iw']
w = decompress(pars['w'], D, initial_wt)

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

    rows = map(readvw, lines)

    y = map(lambda x: 1. if x[0] == '1' else 0., rows)

    X = map(lambda x: get_x(x[1:12], D), rows)

    errorbatch = sum(map(lambda x: 1 if x.has_key('insane') else 0, X))
    errcount += errorbatch

    pr = map(lambda x: get_p(x, w), X)
    p = map(lambda x: x[0], pr)
    #r = map(lambda x: x[1], pr)

    lossb = sum([logloss(p_, y_) for (p_, y_) in zip(p, y)])
    loss += lossb

    t += lenbatch
    print str(datetime.now()) + " Encountered:",t,"Current loss:",lossb/lenbatch,"Total loss:",loss/t
    sys.stdout.flush()

    for i in range(lenbatch):
        outfile.write('%d %f\n' % (int(y[i]), p[i]))

outfile.close()

print 'Error count:',errcount
print ('FINAL AVG LOSS = %f' % (loss/t))

#with open('testscores','a') as f:
#    f.write("alpha "+ comment+ ": " + str(loss/t) + "\n")


#!/usr/bin/pypy

import sys
from datetime import datetime
from csv import DictReader
from functions import logloss, get_x, get_p

test = sys.stdin
modelfile = sys.argv[1]
outfile = sys.argv[2]

logbatch = 100000

import pickle
f = open(modelfile,'r')
pars = pickle.load(f)
f.close()

D = pars['D']
w = pars['w']
header = pars['header']
extra = pars['extra']

# testing (build preds file)
loss = 0.
lossb = 0.
errorcount = 0
with open(outfile, 'w') as preds:
    for t, row in enumerate(DictReader(test, header, delimiter='^')):
        y = 1. if row['clk'] == '1' else 0.

        del row['clk']  # can't let the model peek the answer

        for h in (extra):
            del row[h]

        x = get_x(row, D)
        if x == -1:
            errorcount += 1
            continue

        p, r = get_p(x, w)

        # for progress validation, useless for learning our model
        lossx = logloss(p, y)
        loss += lossx
        lossb += lossx
        
        if t % logbatch == 0 and t > 1:
            print('%s\tencountered: %d\tcurrent whole logloss: %f\tcurrent batch logloss: %f' % (datetime.now(), t, loss/t, lossb/logbatch))
            lossb = 0.

        preds.write('%d %f\n' % (int(y), r))

print 'Error count:',errorcount
print ('FINAL AVG LOSS = %f' % (loss/t))


#with open('testscores','a') as f:
#    f.write("Final : " + str(loss/t) + "\n")


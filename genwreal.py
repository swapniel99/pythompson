import sys, pickle
from numpy import random

b = sys.argv[1]

D = 2 ** int(b)

Wreal = map(float, list(random.normal(-.5, 1, D)))
if Wreal[D - 1] > 0.:
    Wreal[D - 1] *= -1.

dumpfile=open('wreal' + b, 'w')
pickle.dump(Wreal,dumpfile)
dumpfile.close()


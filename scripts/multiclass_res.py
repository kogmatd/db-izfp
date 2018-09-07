#!/usr/bin/python3

import os
import sys
import re
import importlib
import numpy as np

sys.path.append(os.path.join(os.environ['HOME'],'audiomix/anawav'))

import ipl
import icfg
import icls
from ihelp import *
importlib.reload(ipl)
importlib.reload(icfg)
importlib.reload(icls)
importlib.reload(ihelp)

if len(sys.argv)<2: raise ValueError("Usage: "+sys.argv[0]+" CFG prob_*.npy")
icfg.Cfg(sys.argv[1])

ftst=icfg.readflst('test')
dlog=icfg.getdir('log')

lab=ihelp.rle([f['lab'] for f in ftst])
labs=np.array([l[2] for l in lab])

fns=argv2resfns('res_',sys.argv[2:])

for fn in fns:
    cls,fea,s=os.path.basename(fn)[4:-4].split('_')
    res=np.load(fn)
    if cls=='hmm': res=-res
    resc=labs[res.argmax(axis=1)]
    icls.cmp(ftst,resc,'%s %s %s'%(cls,fea,s))

#!/usr/bin/python3

import os
import sys
import re
import importlib
import numpy as np

sys.path.append(os.path.join(os.environ['HOME'],'audiomix/anawav'))

import ipl
import icfg
importlib.reload(ipl)
importlib.reload(icfg)

def rle(x):
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return np.array([], dtype=int)
    starts = np.r_[0, np.where(x[1:]!=x[:-1])[0] + 1]                                                                                                                                                                                                    
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]
    return [(starts[i],lengths[i], values[i]) for i in range(len(starts))]

def plot_roc(ftst,nld):
    okpat='Z00'
    if icfg.get('db')=='izfp/cfk': okpat='Z0[0-2]'
    detect=(
       nld[:,np.array([not re.match(okpat,f['lab']) is None for f in ftst])].mean(axis=0),
       nld[:,np.array([    re.match(okpat,f['lab']) is None for f in ftst])].mean(axis=0),
    )
    dres=[]
    for v in sorted([*detect[0],*detect[1]]):
        dres.append({
            'x':v,
            'tp':np.sum(detect[0]>v),
            'fp':np.sum(detect[1]>v),
            'fn':np.sum(detect[0]<=v),
            'tn':np.sum(detect[1]<=v),
        })
    for r in dres:
        r['facc']=r['fp']/(r['fp']+r['tn'])
        r['frej']=r['fn']/(r['fn']+r['tp'])
    dpl=sorted([[r['facc'],r['frej']] for r in dres],key=lambda a: a[0]*100+(1-a[1]))
    #ipl.p2(dpl,xlabel='Fehlakzeptanz',ylabel='Fehlrückweisung')

    # find eer
    i=np.sum([r[0]<r[1] for r in dpl])
    (ax,ay)=dpl[i-1]
    (bx,by)=dpl[i]
    eer=(by*ax-ay*bx)/(by+ax-ay-bx)
    print('EER: %.2f%%'%(eer*100))
    if eer==0:
        cm=(np.min(detect[0])-np.max(detect[1]))/(np.mean(detect[0])-np.mean(detect[1]))
        print('CM: %.2f%%'%(cm*100))

if len(sys.argv)<2: raise ValueError("Usage: "+sys.argv[0]+" CFG prob_*.npy")
icfg.Cfg(sys.argv[1])

ftst=icfg.readflst('test')
dlog=icfg.getdir('log')

lab=rle([f['lab'] for f in ftst])

fns=sys.argv[2:]
if len(fns)==0: fns=[dlog]
i=0
while i<len(fns):
    if not os.path.exists(fns[i]): fns[i]=os.path.join(dlog,fns[i])
    if os.path.isdir(fns[i]):
        ins=[os.path.join(fns[i],f) for f in os.listdir(fns[i]) if f[:5]=='prob_' and f[-4:]=='.npy']
        fns=fns[:i]+ins+fns[i+1:]
        i+=len(ins)
    else: i+=1

for fn in sorted(fns):
    print('##### %s #####'%(os.path.basename(fn)))
    prob=np.load(fn)

    if len(fns)==1:
        probm=np.mean(prob,axis=0)
        res=[(int(l[2][1:]),probm[l[0]:l[0]+l[1]]) for l in lab]
        resmean=[(l,np.mean(r),np.std(r)) for (l,r) in res]
        ipl.p2(resmean,err='y',xrange=(-1,40))
        #ipl.cm(prob)

    plot_roc(ftst,prob)


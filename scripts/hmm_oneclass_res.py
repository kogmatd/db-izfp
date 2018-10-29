#!/usr/bin/python3

import os
import sys
import re
import importlib
import numpy as np

sys.path.append(os.environ['UASR_HOME']+'-py')

import idlabpro
import icfg
import icls
import ipl
importlib.reload(icfg)
importlib.reload(icls)
importlib.reload(ipl)
importlib.reload(idlabpro)

def loadass():
    dlog=icfg.getdir('log')
    ass=idlabpro.PData()
    nld=None
    ftst=icfg.readflst('test')
    cls=icls.getcls(ftst)
    for fn in sorted(os.listdir(dlog)):
        if not re.match('hmm-0_0_assess\.',fn): continue
        ass.Restore(os.path.join(dlog,fn))
        ass.Select(ass,0,1)
        dat=ass.tonumpy().reshape((1,-1))
        if nld is None: nld=dat
        else: nld=np.concatenate((nld,dat),axis=0)
    if len(ftst)!=nld.shape[-1]: raise ValueError("test.flst and assess missmatch in len")
    return (ftst,nld)

def plot_mean_nll(ftst,nld):
    res=[]
    for c in cls:
        nld1=nld[:,np.array([f['lab']==c for f in ftst])]
        res.append([int(c[1:]),nld1])

    ##ipl.p2([[r[0],r[1].mean(),r[1].std()] for r in res],err='y',xlabel='Risslänge in cm',ylabel='mean(NLL), std(NLL)')
    ##ipl.p2([[r[0],r[1].mean(),r[1].std(axis=1).mean()] for r in res],err='y',xlabel='Risslänge in cm',ylabel='mean(NLL), mean(std(NLL))')
    ##ipl.p2(np.transpose([[r[0]*np.ones(r[1].shape[0]),r[1].mean(axis=1),r[1].std(axis=1)] for r in res],(2,0,1)),err='y',xlabel='Risslänge in cm',ylabel='mean(NLL), std(NLL)')

    ipl.p2([[r[0],r[1].mean(axis=0).mean(),r[1].mean(axis=0).std()] for r in res],err='y',xlabel='Risslänge in cm',ylabel='mean(NLL), std(mean(NLL))')

def plot_roc(ftst,nld):
    detect=(
       nld[:,np.array([f['lab']=='Z00' for f in ftst])].mean(axis=0),
       nld[:,np.array([f['lab']!='Z00' for f in ftst])].mean(axis=0),
    )
    dres=[]
    for v in sorted([*detect[0],*detect[1]]):
        dres.append({
            'x':v,
            'tp':np.sum(detect[0]<v),
            'fp':np.sum(detect[1]<v),
            'fn':np.sum(detect[0]>=v),
            'tn':np.sum(detect[1]>=v),
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
        cm=(np.min(detect[1])-np.max(detect[0]))/(np.mean(detect[1])-np.mean(detect[0]))
        print('CM: %.2f%%'%(cm*100))



if len(sys.argv)<2: raise ValueError("Usage: "+sys.argv[0]+" CFG")
icfg.Cfg(sys.argv[1])

(ftst,nld)=loadass()
plot_mean_nll(ftst,nld)
plot_roc(ftst,nld)


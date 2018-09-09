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
import ihelp
from ihelp import *
importlib.reload(ipl)
importlib.reload(icfg)
importlib.reload(icls)
importlib.reload(ihelp)

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

def hmmprob(nld):
    px=nld.take(0,-1)
    pxs=sorted(px.flat)
    l=pxs[int(len(pxs)*0.05)]
    h=pxs[int(len(pxs)*0.95)]
    v=np.exp(-(px-l)/(h-l))
    v1=np.exp(-1)
    return (v-v1)/(1-v1)*0.8+0.1
    


if len(sys.argv)<2: raise ValueError("Usage: "+sys.argv[0]+" CFG prob_*.npy")
icfg.Cfg(sys.argv[1])

ftst=icfg.readflst('test')
dlog=icfg.getdir('log')

lab=ihelp.rle([f['lab'] for f in ftst])

fns=argv2resfns('prob_',sys.argv[2:])

for fn in sorted(fns):
    prob=np.load(fn)
    msg=''
    if fn[-8:]=='_old.npy': continue
    if fn.find('_hmm_')>=0 and prob.shape[-1]==3:
        if np.sum(prob.take(2,-1)!=prob.take(2,-1)[0])==0: msg=' ERR in hmm[1]'
        prob=hmmprob(prob.take([1,2],-1))
        #prob=prob.take(0,-1)

    okpat='Z0[0-2]' if icfg.get('db')=='izfp/cfk' else 'Z00'
    #okpat='Z0[01]'
    eer,cm=icls.eer(prob,flst=ftst,okpat=okpat)
    
    #ref=[not re.match(okpat,f['lab']) is None for f in ftst]
    #msg+=' [%.2f<=>%.2f]'%(np.mean(prob.mean(axis=0)[ref]),np.mean(prob.mean(axis=0)[np.array(ref)==False]))
    if np.max(prob.mean(0))<0.4: msg+=' ERR low max %.2f'%(np.max(prob.mean(0)))
    if np.min(prob.mean(0))>0.6: msg+=' ERR high min %.2f'%(np.min(prob.mean(0)))

    print('%-20s EER: %6.2f%% CM: %6.2f%%%s'%(os.path.basename(fn),eer*100,cm*100,msg))

    if len(fns)==1:
        probm=np.mean(prob,axis=0)
        res=[(int(l[2][1:]),probm[l[0]:l[0]+l[1]]) for l in lab]
        resmean=[(l,np.mean(r),np.std(r)) for (l,r) in res]
        ipl.p2(resmean,err='y',xrange=(-1,resmean[-1][0]+1))
        #ipl.cm(prob)

    #plot_roc(ftst,prob)


#!/usr/bin/python3

import os
import sys
import re
import importlib
import numpy as np

sys.path.append(os.environ['UASR_HOME']+'-py')

import ipl
import icfg
import icls
import ihelp
from ihelp import *

def plot_roc(ftst,nld):
    oklab = list(set(map(lambda x: x['lab'], ftst)))
    oklab.sort()
    okpat=oklab[0]
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
    ipl.p2(dpl,xlabel='Fehlakzeptanz',ylabel='Fehlrückweisung')

def hmmprob(nld):
    px=nld.take(0,-1)
    pxs=sorted(px.flat)
    l=pxs[int(len(pxs)*0.05)]
    h=pxs[int(len(pxs)*0.95)]
    v=np.exp(-(px-l)/(h-l))
    v1=np.exp(-1)
    return (v-v1)/(1-v1)*0.8+0.1
    
if len(sys.argv)<2: raise ValueError("Usage: "+sys.argv[0]+" CFG prob_*.npy")
#if sys.argv[1]=='csv':
#    for cfgi in range(0,4):
#        if cfgi==0: cfg='../als/oneclass/info/default.cfg'
#        else: cfg='../cfk/X%i-oneclass/info/default.cfg'%cfgi
#        os.system(str.join(' ',['python3',sys.argv[0],cfg,'csv']))
#    raise SystemExit()
icfg.Cfg(sys.argv[1])
ocsv=len(sys.argv)>2 and sys.argv[2]=='csv'
if ocsv: sys.argv.remove('csv')
fns=sys.argv[2:]

ftst=icfg.readflst('test')
dlog=icfg.getdir('log')

fns=argv2resfns('prob_',fns)

labmap={}
labmap=getlabmaps()
if not icfg.get('labmap') is None: labmap=icfg.get('labmap'); labmap=eval(labmap)
if labmap is not None:
    ftst.maplab(labmap)
lcls=np.array(ftst.getcls())

out=[]
for fn in sorted(fns):
    prob=np.load(fn)
    msg=''
    if fn[-8:]=='_old.npy': continue
    if fn.find('_hmm_')>=0 and prob.shape[-1]==3:
        if np.sum(prob.take(2,-1)!=prob.take(2,-1)[0])==0: msg=' ERR in hmm[1]'
        prob=hmmprob(prob.take([1,2],-1))
        #prob=prob.take(0,-1)

    okpat=lcls[0]
    eer,cm=icls.eer(prob,flst=ftst,okpat=okpat)
    
    #ref=[not re.match(okpat,f['lab']) is None for f in ftst]
    #msg+=' [%.2f<=>%.2f]'%(np.mean(prob.mean(axis=0)[ref]),np.mean(prob.mean(axis=0)[np.array(ref)==False]))
    if np.max(prob.mean(0))<0.4: msg+=' ERR low max %.2f'%(np.max(prob.mean(0)))
    if np.min(prob.mean(0))>0.6: msg+=' ERR high min %.2f'%(np.min(prob.mean(0)))

    if ocsv:
        (cls,fea)=os.path.basename(fn)[5:-4].split('_')
        out.append('"%s" "%s" "%s" %s %s "%s"'%(
            icfg.get('db').split('/')[-1]+'/'+icfg.get('exp'),
            fea,cls,
            '%.1f%%'%(eer*100) if eer!=0 else '---',
            '%.1f%%'%(cm*100)  if eer==0 else '---',
            msg[1:]
        ))
    else: print('%-20s EER: %6.2f%% CM: %6.2f%%%s'%(os.path.basename(fn),eer*100,cm*100,msg))

    #if len(fns)==1:
        #plt_clsmeanstd(prob.mean(axis=0),ftst)
        #ipl.cm(prob)

    #plot_roc(ftst,prob)

lo=None
for o in sorted(out):
    to=o.split(' ')[:2]
    if not lo is None and lo!=to: print('')
    lo=to
    print(o)
print('')

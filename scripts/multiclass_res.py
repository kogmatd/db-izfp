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

if len(sys.argv)<2: raise ValueError("Usage: "+sys.argv[0]+" CFG prob_*.npy")
icfg.Cfg(sys.argv[1])

ftst=icfg.readflst('test')
dlog=icfg.getdir('log')

okpat='Z0[0-2]' if icfg.get('db')=='izfp/cfk' else 'Z00'
for f in ftst:
    if re.match(okpat,f['lab']): f['lab']='Z00'

lab=ihelp.rle([f['lab'] for f in ftst])
labs=np.array([l[2] for l in lab])

fns=argv2resfns('res_',sys.argv[2:])

lcls=icls.getcls(ftst)

resh={}
for fn in fns:
    cls,fea,s=os.path.basename(fn)[4:-4].split('_')
    res=np.load(fn)
    if cls=='hmm': res=-res
    resc=labs[res.argmax(axis=-1)]
    c=np.sum(resc==[f['lab'] for f in ftst])/len(ftst)
    if not cls in resh: resh[cls]={}
    if not fea in resh[cls]: resh[cls][fea]={}
    if s in resh[cls][fea]: raise ValueError("duplicate")
    resh[cls][fea][s]={
        'res':res,
        'resc':resc,
        'c':c,
    }

for cls in resh:
    for fea in resh[cls]:
        c=[]
        s=[]
        res=[]
        for si,ri in resh[cls][fea].items(): s.append(si); c.append(ri['c']); res.append(ri['res'])
        cmax=np.max(c)
        smax=np.array(s)[np.argmax(c)]
        res=np.mean(res,axis=0)
        resc=labs[res.argmax(axis=-1)]
        cmix=np.sum(resc==[f['lab'] for f in ftst])/len(ftst)
        if cmix<1:
            cmx={}
            for lref in lcls:
                cmx[lref]={}
                for lres in lcls:
                    cmx[lref][lres]=np.sum(resc[np.array([f['lab']==lref for f in ftst])]==lres)
                    cmx[lref][lres]/=np.sum([f['lab']==lref for f in ftst])
            cmxa=[(cmx[lref][lres]*100,lref,lres) for lref in lcls for lres in lcls if lres!=lref]
            cmxamax=cmxa[np.array(cmxa)[:,0].argmax()]
            msg=' MAX-ER: %5.1f%% (%s=>%s)'%cmxamax
        else:
            cma=[]
            for l1i in range(len(lcls)):
                for l2i in range(l1i+1,len(lcls)):
                    if l1i==l2i: continue
                    l1=lcls[l1i]
                    l2=lcls[l2i]
                    r=res[:,l1i]-res[:,l2i]
                    r1=r[np.array([f['lab']==l1 for f in ftst])]
                    r2=r[np.array([f['lab']==l2 for f in ftst])]
                    r1m=r1.mean()
                    r2m=r2.mean()
                    if r1m>r2m: cm=(r1.min()-r2.max())/(r1m-r2m)
                    else: cm=(r2.min()-r1.max())/(r2m-r1m)
                    cma.append((cm*100,l1,l2))
            cmmin=cma[np.array(cma)[:,0].argmin()]
            msg=' MAX-ER: 0 MIN-CM %5.1f%% (%s<=>%s)'%cmmin
        print('%s %s [%3i] best %5.1f%%/%s mix %5.1f%%%s'%(cls,fea,len(resh[cls][fea]),cmax*100,smax,cmix*100,msg))

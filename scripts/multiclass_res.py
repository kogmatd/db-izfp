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

def cor(x,y):
    a=x-np.mean(x)
    b=y-np.mean(y)
    if np.max(np.abs(a))==0: return 0
    if np.max(np.abs(b))==0: return 0
    return np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))

def mse(x,y):
    return np.mean((x-y)**2)

if len(sys.argv)<2: raise ValueError("Usage: "+sys.argv[0]+" (CFG prob_*.npy | csv)")
if sys.argv[1]=='csv':
    for cfgi in range(1,4):
        cfg='../cfk/X%i/info/default.cfg'%cfgi
        os.system(str.join(' ',['python3',sys.argv[0],cfg,'csv']))
    raise SystemExit()
icfg.Cfg(sys.argv[1])
ocsv=len(sys.argv)>2 and sys.argv[2]=='csv'
if ocsv: sys.argv.remove('csv')
fns=sys.argv[2:]

ftst=icfg.readflst('test')
dlog=icfg.getdir('log')

okpat='Z0[0-2]' if icfg.get('db')=='izfp/cfk' else 'Z00'
for f in ftst:
    if re.match(okpat,f['lab']): f['lab']='Z00'

lab=ihelp.rle([f['lab'] for f in ftst])
labs=np.array([l[2] for l in lab])

fns=argv2resfns('res_',fns)

lcls=icls.getcls(ftst)

resh={}
for fn in fns:
    cls,fea,s=os.path.basename(fn)[4:-4].split('_')
    res=np.load(fn)
    if res.shape==(0,): res=np.zeros((len(ftst),1 if icfg.get('trn.regression')==True else len(lcls)))
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

feas=sorted({fea for rc in resh.values() for fea in rc.keys()})
sens=sorted({s for rc in resh.values() for rf in rc.values() for s in rf.keys()})

#feas=['pfa']

for fea in feas:
    for cls in sorted(resh):
        if not fea in resh[cls]: continue
        c=[]
        s=[]
        res=[]
        for si,ri in resh[cls][fea].items(): s.append(si); c.append(ri['c']); res.append(ri['res'])
        if icfg.get('trn.regression'):
            res=np.array(res)
            res=res.reshape(res.shape[:-1])
            ressel=res[np.min(res,1)!=np.max(res,1)]
            mres=np.mean(ressel,axis=0)
            lab=np.array([float(f['lab'][1:]) for f in ftst])
            lfcls=list(map(lambda l:float(l[1:]), lcls))
            acor=cor(mres,lab)
            amse=mse(mres,lab)
            lmse=[mse(mres[lab==l],l) for l in lfcls]
            print('%s %-7s [%3i/%3i] MEAN cor: %.3f mse: %5.2f max-mse: %5.2f/%s Z00/1-mse: %5.2f, %5.2f'%(
                fea,cls,len(resh[cls][fea]),len(ressel),
                acor,amse,np.max(lmse),lcls[np.argmax(lmse)],*lmse[:2]
            ))
            bcor=[cor(r,lab) for r in res]
            bmse=[mse(r,lab) for r in res]
            #print('%s %-7s [%3i/%3i] BEST cor: %.3f/%s mse: %5.2f/%s'%(
            #    fea,cls,len(resh[cls][fea]),len(ressel),
            #    np.max(bcor),str(np.argmax(bcor)),
            #    np.max(bmse),str(np.argmax(bmse)),
            #))
            #ipl.p2((lab,mres),nox=True,title=fea+' '+cls)
            #ipl.p2(res,nox=True,title=fea+' '+cls)
            #raise SystemExit()
        else:
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
                cmxmax=np.max([cmx[lref][lres] for lref in lcls for lres in lcls if lres!=lref])
                cmxmaxl=str.join(' ',[lref+'=>'+lres for lref in lcls for lres in lcls if lres!=lref and cmx[lref][lres]==cmxmax])
                msg=' MAX-ER: %5.1f%% (%s)'%(cmxmax*100,cmxmaxl)
                cmmin=None
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
                msg=' MAX-ER:   0.0%% MIN-CM %5.1f%% (%s<=>%s)'%cmmin
                cmxmax=0
                cmxmaxl='---'
            misstrain=len([1 for ri in resh[cls][fea].values() if ri['res'].max()==ri['res'].min()])
            if misstrain>0: msg+=' MISSTRAIN: %i'%(misstrain)
            if ocsv:
                print('"%s" "%s" "%s" %s "%s" %s %s "%s" %.1f%% "%s"'%(
                    icfg.get('db').split('/')[-1]+'/'+icfg.get('exp'),
                    fea,cls,
                    '%.1f%%'%((1-cmix)*100) if len(resh[cls][fea])==len(sens) else '"RUN"',
                    cmxmaxl               if cmxmaxl!='---' else '%s<=>%s'%(cmmin[1:]),
                    '%.1f%%'%(cmxmax*100) if cmxmaxl!='---' else '"---"',
                    '%.1f%%'%(cmmin[0])   if cmxmaxl=='---' else '"---"',
                    smax,(1-cmax)*100,
                    'MISSTRAIN: %i'%(misstrain) if misstrain>0 else '',
                ))
            else:
                print('%s %4s [%3i] best %5.1f%%/%s mix %5.1f%%%s'%(
                    fea,cls,len(resh[cls][fea]),
                    cmax*100,smax,
                    cmix*100,msg
                ))
    print('')

#resmat=np.array([[(resh[cls][fea][si]['c'] if si in resh[cls][fea] else 0) for si in sens] for fea in feas for cls in sorted(resh) if fea in resh[cls]])
#ipl.p2(resmat,nox=True)
#ipl.p2(resmat[:,np.argsort(resmat.mean(axis=0))])


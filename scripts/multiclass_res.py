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
import idat
import ilinreg
from ihelp import *
importlib.reload(ipl)
importlib.reload(icfg)
importlib.reload(icls)
importlib.reload(ihelp)
importlib.reload(idat)
importlib.reload(ilinreg)

def cor(x,y):
    a=x-np.mean(x)
    b=y-np.mean(y)
    if np.max(np.abs(a))==0: return 0
    if np.max(np.abs(b))==0: return 0
    return np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))

def mse(x,y):
    return np.mean((x-y)**2)

def plot_reg(lab,mres,name):
    fn='res/reg/'+name+'.plot'
    labc=rle(lab)
    pl=[(l,np.mean(mres[s:s+e]),np.std(mres[s:s+e])) for s,e,l in labc]
    with open(fn,'w') as f:
        f.write('#-title '+name+'\n')
        f.write('#-xlabel Referenz\n')
        f.write('#-ylabel Ergebnis\n')
        f.write('#-col 2\n')
        f.write('#-typ yerrorlines\n')
        for p in pl: f.write('%i %.3f %.3f\n'%p)


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
ftrn=icfg.readflst('train')
dlog=icfg.getdir('log')
sen=getsensors()
regression=icfg.get('trn.regression')==True

okpat='Z0[0-2]' if icfg.get('db')=='izfp/cfk' else 'Z00'
for f in ftst:
    if re.match(okpat,f['lab']): f['lab']='Z00'

lab=ihelp.rle([f['lab'] for f in ftst])
labs=np.array([l[2] for l in lab])

fns=argv2resfns('res_',fns)

lcls=icls.getcls(ftst)
if regression:
    icls.labf(ftst)
    lab=np.array([f['labf'] for f in ftst])
    lfcls=[i[-1] for i in rle(sorted(lab))]

resh={}
for fn in fns:
    if fn.find('_trn.npy')>=0 or fn.find('.model')>=0: continue
    cls,fea,s=os.path.basename(fn)[4:-4].split('_')
    res=np.load(fn)
    if res.shape==(0,): continue
    h={'res':res}
    if os.path.exists(fn[:-4]+'_trn.npy'): h['res_trn']=np.load(fn[:-4]+'_trn.npy')
    if regression:
        h['c']=0
    else:
        if cls=='hmm': res=-res
        h['resc']=resc=labs[res.argmax(axis=-1)]
        h['c']=np.sum(resc==[f['lab'] for f in ftst])/len(ftst)
    if not cls in resh: resh[cls]={}
    if not fea in resh[cls]: resh[cls][fea]={}
    if s in resh[cls][fea]: raise ValueError("duplicate")
    resh[cls][fea][s]=h

feas=sorted({fea for rc in resh.values() for fea in rc.keys()})
sens=sorted({s for rc in resh.values() for rf in rc.values() for s in rf.keys()})

#feas=['pfa']

for cls,feah in list(resh.items()):
    for fea,senh in feah.items():
           res_trn=[h['res_trn'].flatten() for h in senh.values() if 'res_trn' in h]
           if len(res_trn)<10: continue
           res_tst=[h['res'].flatten() for h in senh.values() if 'res_trn' in h]
           res_trn=np.array(res_trn).transpose()
           res_tst=np.array(res_tst).transpose()
           ftrnc=[{'lab':f['lab'],'fea':idat.Dat(rt)} for f,rt in zip(ftrn,res_trn)]
           icls.labf(ftrnc)
           ftstc=[{'fea':idat.Dat(rt)} for rt in res_tst]
           lr=ilinreg.trn(ftrnc)
           lres=ilinreg.evl(lr,ftstc)
           if not cls+'_lr' in resh: resh[cls+'_lr']={}
           resh[cls+'_lr'][fea]={'XX':{'res':lres,'c':0}}
           

for fea in feas:
    for cls in sorted(resh):
        if not fea in resh[cls]: continue
        c=[]
        s=[]
        res=[]
        for si,ri in resh[cls][fea].items(): s.append(si); c.append(ri['c']); res.append(ri['res'])
        if regression:
            res=np.array(res)
            if res.shape[-1]==1: res=res.reshape(res.shape[:-1])
            mres=np.mean(res,axis=0)
            acor=cor(mres,lab)
            amse=mse(mres,lab)
            lmse=[mse(mres[lab==l],l) for l in lfcls]
            eer,cm=icls.eer(1-mres,flst=ftst,okpat=okpat)
            print('%s %-7s [%3i/%3i] MEAN cor: %.3f mse: %5.2f max-mse: %5.2f/%s Z00/1-mse: %5.2f, %5.2f EER: %6.2f%% CM: %6.2f%%'%(
                fea,cls,len(resh[cls][fea]),len(sen),
                acor,amse,np.max(lmse),lcls[np.argmax(lmse)],*lmse[:2],
                eer*100,cm*100,
            ))
            bcor=[cor(r,lab) for r in res]
            bmse=[mse(r,lab) for r in res]
            #print('%s %-7s [%3i/%3i] BEST cor: %.3f/%s mse: %5.2f/%s'%(
            #    fea,cls,len(resh[cls][fea]),len(sen),
            #    np.max(bcor),str(np.argmax(bcor)),
            #    np.max(bmse),str(np.argmax(bmse)),
            #))
            #ipl.p2((lab,mres),nox=True,title=fea+' '+cls)
            #ipl.p2(res,nox=True,title=fea+' '+cls)
            #raise SystemExit()
            plot_reg(lab,mres,fea+'_'+cls)
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

#plt_clsmeanstd(np.array([r['res'] for r in resh['hmm']['pfa'].values()]),ftst)
#plt_clsmeanstd(np.array([r['res'] for r in resh['hmm']['pfa'].values()]).mean(0),ftst)

#resmat=np.array([[(resh[cls][fea][si]['c'] if si in resh[cls][fea] else 0) for si in sens] for fea in feas for cls in sorted(resh) if fea in resh[cls]])
#ipl.p2(resmat,nox=True)
#ipl.p2(resmat[:,np.argsort(resmat.mean(axis=0))])


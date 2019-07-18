#!/usr/bin/python3
import os
import numpy as np
import sys
import re
import time
import itertools
import importlib

sys.path.append(os.environ['UASR_HOME']+'-py')

import ipl
import isig
import ifea
import icfg
import isvm
import ihmm
import idnn
import iktf
import icls
import idat
import ijob
import ifdb
import ihelp
from ihelp import *

def prtres(prob,flst):
    oklab = list(set(map(lambda x: x['lab'],flst)))
    oklab.sort()
    eer,cm = icls.eer(prob,flst=flst,okpat=oklab[0]) #good label
    return 'EER: %6.2f%% CM: %6.2f%%'%(eer*100,cm*100)

def svmtrn(ftrn,ftst,fea,s,kwargs={}):
    print('svm start  '+s)
    csvm=isvm.trn(ftrn,fea,**kwargs)
    prob=isvm.evlp(csvm,ftst,fea)[:,0]
    print('svm finish '+s+' '+prtres(prob,ftst))
    return prob

def hmmtrn(ftrn,ftst,fea,s,kwargs={}):
    print('hmm start  '+s)
    chmm=ihmm.trn(flst=ftrn,fea=fea,its=[0],states=3)
    nld=ihmm.evlp(chmm,flst=ftst,fea=fea)
    prob=-nld.take(0,-1)/np.array([f[fea].shape[0] for f in ftst])
    print('hmm finish '+s+' '+prtres(prob,ftst))
    return np.concatenate((prob.reshape(-1,1),nld),axis=1)
    #return prob

def ktftrn(ftrn,ftst,fea,s,kwargs={}):
    print('dnn start  '+s)
    ktf = iktf.ModKeras(**kwargs)
    ktf.trn(ftrn, fea)
    prob = ktf.evl(ftst, fea, prob=True)[:,0]
    # probability of class 0
    prob = 1-prob
    print('ktf stop  '+s+' '+prtres(prob, ftst))
    return prob

def dnntrn(ftrn,ftst,fea,s,kwargs={}):
    print('dnn start  '+s)
    cdnn=idnn.trn(ftrn,ftst,fea=fea,dmod=dmod,**kwargs)
    prob=idnn.evlp(cdnn,ftst,fea=fea)[:,0]
    print('dnn finish '+s+' '+prtres(prob,ftst))
    return prob

def dnntest(s,fea='sfa',**kwargs):
    ftst=ftsts[s]
    ftrn=ftrns[s]
    ftst2=[{**f} for f in ftst]
    for f in ftst2:
        if f['lab']!='Z00': f['lab']='Zxx'
    cdnn=idnn.trn(ftrn,ftst2,fea=fea,dmod=dmod,**kwargs)
    res=idnn.evl(cdnn,ftst2,fea=fea)
    lab=[f['lab'] for f in ftst2]
    c=len([True for l,r in zip(lab,res) if l==r])
    print(c,len(lab),c/len(lab))
    return cdnn

import random
def dnnrnd():
    xip=int((random.random()**2)*1000)+10
    dp=(random.random()**5)*0.5
    kwargs={}
    kwargs['max_iter']=200
    kwargs['s']=sen[int(random.random()*len(sen))]
    kwargs['fea']=['sig','pfa','sfa'][int(random.random()*3)]
    kwargs['weight_decay']=(random.random()**4)*0.4
    kwargs['base_lr']=(random.random()**2)*0.1+0.001
    lay=[('ip',xip),('dropout',dp),('relu',),('ip',)]
    if True:
        k=int(random.random()*5)+3
        o=int(random.random()*15)+10
        s=int(random.random()*3)+1
        if kwargs['fea']=='sig':
            k=[k,1]
            s=[s,1]
        lay=[('conv',k,o,s)]+lay
    kwargs['lay']=lay
    # c 16,7,2,0,1 # ip 498
    #kwargs={'fea': 'pfa', 's': 'A1A2', 'max_iter': 200, 'weight_decay': 0.10661077392254943, 'lay': [('conv', 3, 18, 1), ('ip', 824), ('dropout', 0.00038962239039409444), ('relu',), ('ip',)], 'base_lr': 0.08700124692413123}
    stat=[]
    for i in range(3):
        print(kwargs)
        r=dnntest(**kwargs)
        stat.append(np.array(r['stat']))
    print('/dnnrnd_'+str(int(time.time())))
    np.save(dlog+'/dnnrnd/dnnrnd_'+str(int(time.time()))+'_arg.npy',kwargs)
    np.save(dlog+'/dnnrnd/dnnrnd_'+str(int(time.time()))+'_stat.npy',stat)

def dnnrndloop():
    while not os.path.exists('stop'): dnnrnd()

def dnnrnd_allsen():
    dn=os.path.join(dlog,'dnnrnd')
    for fn in os.listdir(dn):
        if os.path.exists('stop'): break
        if fn[-8:]!='_arg.npy': continue
        fna=os.path.join(dn,fn)
        fns=fna[:-8]+'_stat.npy'
        fnx=fna[:-8]+'_stat_allsen.npy'
        if not os.path.exists(fns): continue
        if os.path.exists(fnx): continue
        stat=np.load(fns)
        best=np.max(np.min(np.array(stat)[:,:,2],axis=0))
        if best<0.8: continue
        kwargs=eval(str(np.load(fna)))
        statx=[]
        for s in sen:
            kwargs['s']=s
            print(kwargs)
            r=dnntest(**kwargs)
            statx.append(np.array(r['stat']))
        np.save(fnx,statx)

if len(sys.argv)<2: raise ValueError("Usage: "+sys.argv[0]+" CFG [-n]")
icfg.Cfg(*sys.argv[1:])

if '-nn' in sys.argv: raise SystemExit()

print("flst")
ftrn=icfg.readflst('train')
ftst=icfg.readflst('test')
#fdev=[] if icfg.get('flist.dev') is None else icfg.readflst('dev')

dmod=icfg.getdir('model')
if not os.path.exists(dmod): os.mkdir(dmod)
dlog=icfg.getdir('log')
if not os.path.exists(dlog): os.mkdir(dlog)
dsig=icfg.getdir('sig')
sigext='.'+icfg.get('sig.ext','wav')
sen=getsensors()

senuse=sen
feause=['pfa','sfa','sig']
clsuse=['hmm','svm','cnnb','cnn','dnn','ktf']
if not icfg.get('senuse') is None: senuse=icfg.get('senuse').split(',')
if not icfg.get('feause') is None: feause=icfg.get('feause').split(',')
if not icfg.get('clsuse') is None: clsuse=icfg.get('clsuse').split(',')

labmap={}
labmap=getlabmaps()
if not icfg.get('labmap') is None: labmap=icfg.get('labmap'); labmap=eval(labmap)

maxjobs=16

# the first label is of interest
if labmap is not None:
    ftrn = ftrn.maplab(labmap)
    ftst = ftst.maplab(labmap)

if senuse is not None:
    ftrns = {}
    for strn in senuse:
        ftrns[strn] = []
        ftrns[strn] = ftrn.expandsensor(sen)
        for f in ftrns[strn]: f['lab'] = labmap[list(labmap.keys())[0]] if f['sen']==strn else labmap[list(labmap.keys())[1]]
        ftrns[strn] = ftrns[strn].equalcls()
    ftsts = {s: ftst.expandsensor(s) for s in senuse}
else:
    sen=senuse = [icfg.get('db')]
    ftrns = {}
    ftsts = {}
    ftrns[sen[0]] = ftrn
    ftsts[sen[0]] = ftst

if not 'fdb' in locals():
    fdb = ifdb.Fdb()
else:
    fdb.chg=False

for typ in ['sig','pfa']:
    print(typ)
    fdb.analyse(typ, eval(typ+'get'), flst=sum(ftrns.values(),[])+sum(ftsts.values(),[]), jobs=maxjobs)
fdb.save()

print('sfa')
do=set(s for s in senuse if any(not 'sfa' in f for f in ftrns[s]+ftsts[s]))
if len(do)==1 or maxjobs==1:
    for s in do: sfaget(ftrns[s],ftsts[s],fdb)
else:
    thr=ijob.Thr(maxjobs)
    for s in do:
        if os.path.exists('stop'): thr.cleanup(); raise SystemExit()
        print('sfa start  '+s)
        thr.start('sfa_'+s,sfaget,(ftrns[s],ftsts[s],fdb))
    for s in do: thr.res('sfa_'+s)

if '-n' in sys.argv: raise SystemExit()
#dnnrndloop()
#raise SystemExit()

#maxjobs=1
print('cls')
for cls in clsuse:
    for fea in feause:
        if cls=='hmm' and ftrns[senuse[0]][0][fea].shape[-1] > 40: continue
        probfn=os.path.join(dlog,'prob_'+cls+'_'+fea+'.npy')
        #if os.path.exists(probfn): continue
        kwargs=icfg.get('trnargs.%s.%s'%(cls,fea))
        if kwargs is None: kwargs={}
        else:
            print('trnargs = '+kwargs)
            kwargs=eval(kwargs)
        fnctrn=eval(cls+'trn')
        if len(senuse)==1 or maxjobs==1 or cls=='dnn' or cls=='ktf':
            prob=[fnctrn(ftrns[s],ftsts[s],fea,s,kwargs) for s in senuse]
        else:
            job=ijob.Thr(maxjobs)
            for s in senuse:
                if os.path.exists('stop'): job.cleanup(); raise SystemExit()
                job.start(cls+'trn_'+s,fnctrn,(ftrns[s],ftsts[s],fea,s,kwargs))
            prob=[job.res(cls+'trn_'+s) for s in senuse]
        if senuse==sen: np.save(probfn,prob)

#x=dnntest('A1A2','sig',max_iter=200,lay=[('conv',[7,1],20,[5,1]),('ip',200),('dropout',0.5),('relu',),('ip',)])
#x=dnntest('A1A2','pfa',max_iter=200,lay=[('ip',20),('relu',),('ip',)],weight_decay=0.1)
#x=dnntest('A1A2','pfa',max_iter=200,lay=[('ip',200),('dropout',0.5),('relu',),('ip',)])
#x=dnntest('A1A2','pfa',max_iter=200,lay=[('conv',7,20,5),('ip',200),('dropout',0.5),('relu',),('ip',)],weight_decay=0.01)

# %bg _ip.magic('run -i foo.py')

raise SystemExit()


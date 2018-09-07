#!/usr/bin/python3

import os
import sys
import re
import time
import importlib
import numpy as np

sys.path.append(os.path.join(os.environ['HOME'],'audiomix/anawav'))

import ipl
import isig
import ifea
import icfg
import isvm
import ihmm
import idnn
import icls
import idat
import ijob
import ifdb
import ihelp
from ihelp import *
importlib.reload(ipl)
importlib.reload(isig)
importlib.reload(ifea)
importlib.reload(icfg)
importlib.reload(isvm)
importlib.reload(ihmm)
importlib.reload(idnn)
importlib.reload(icls)
importlib.reload(idat)
importlib.reload(ijob)
importlib.reload(ifdb)
importlib.reload(ihelp)

def svmtrn(ftrn,ftst,fea,s,kwargs={}):
    print('svm start  '+s)
    #if not 'C' in kwargs: kwargs['C']=1
    #if not 'tol' in kwargs: kwargs['tol']=0.1
    csvm=isvm.trn(ftrn,fea,**kwargs)
    prob=isvm.evlp(csvm,ftst,fea)
    print('svm finish '+s+' ')
    return prob

def hmmtrn(ftrn,ftst,fea,s,kwargs={}):
    print('hmm start  '+s)
    chmm=ihmm.trn(flst=ftrn,fea=fea,its=[0],states=3)
    nld=ihmm.evlp(chmm,flst=ftst,fea=fea)
    res=np.array(chmm['cls'])[nld.argmin(axis=1)]
    icls.cmp(ftst,res,'hmm finish '+s)
    return nld

def dnntrn(ftrn,ftst,fea,s,kwargs={}):
    print('dnn start  '+s)
    cdnn=idnn.trn(ftrn,ftst,fea=fea,dmod=dmod,**kwargs)
    if cdnn is None:
        print('dnn failed '+s)
        return []
    else:
        prob=idnn.evlp(cdnn,ftst,fea=fea)
        res=np.array(cdnn['cls'])[prob.argmax(axis=1)]
        icls.cmp(ftst,res,'dnn finish '+s)
        return prob
cnntrn=dnntrn

if len(sys.argv)<2: raise ValueError("Usage: "+sys.argv[0]+" CFG [-n]")
icfg.Cfg(sys.argv[1])

ftrn=icfg.readflst('train')
ftst=icfg.readflst('test')
#fdev=[] if icfg.get('flist.dev') is None else icfg.readflst('dev')

dmod=icfg.getdir('model')
dlog=icfg.getdir('log')
dsig=icfg.getdir('sig')
sigext='.'+icfg.get('sig.ext','wav')
sen=getsensors()

senuse=sen[:1]

#clsuse=['hmm']
#clsuse=['hmm','svm']
#clsuse=['dnn']
clsuse=['cnn','dnn']

#feause=['sfa']
feause=['sig','pfa','sfa']

okpat='Z0[0-2]' if icfg.get('db')=='izfp/cfk' else 'Z00'

if len(sys.argv)>2 and sys.argv[2]=='-nn': raise SystemExit()

def run_sen(s):

    print("flst [%s]"%(s))
    ftrns=flstexpandsen(ftrn,s,okpat)
    ftsts=flstexpandsen(ftst,s,okpat)
    ftrns=icls.equalcls(ftrns)

    fdb=ifdb.load(s)

    for f in ftrns+ftsts:
        if not f['fn'] in fdb: fdb[f['fn']]={'fn':f['fn']}

    fdb_chg=False
    for typ in ['sig','pfa']:
        do=[f for f in fdb.values() if not typ in f]
        if len(do)==0: continue
        print('%s [%s]'%(typ,s))
        fdb_chg=True
        fnc=eval(typ+'get')

        t=time.time()
        for i in range(len(do)):
            if time.time()-t>5:
                print("%s [%s] %i/%i"%(typ,s,i,len(do)))
                t=time.time()
            fnc([do[i]])

    fealnk(ftrns+ftsts,fdb)

    if len([True for f in ftrns+ftsts if not 'sfa' in f])!=0:
        print('sfa [%s]'%(s))
        sfaget(ftrns,ftsts)
        for f in ftrns+ftsts: fdb[f['fn']]['sfa']=f['sfa']

    if fdb_chg: ifdb.save(fdb,s)

    if len(sys.argv)>2 and sys.argv[2]=='-n': return

    for cls in clsuse:
        for fea in feause:
            if cls=='hmm' and ftrns[0][fea].shape[-1]>40: continue
            resfn=os.path.join(dlog,'res_'+cls+'_'+fea+'_'+s+'.npy')
            if os.path.exists(resfn): continue
            kwargs=icfg.get('trnargs.%s.%s'%(cls,fea))
            if kwargs is None:
                if cls=='dnn' or cls=='cnn': continue
                kwargs={}
            else:
                print('trnargs = '+kwargs)
                kwargs=eval(kwargs)
            fnctrn=eval(cls+'trn')
            res=fnctrn(ftrns,ftsts,fea,s,kwargs)
            np.save(resfn,res)

if len(senuse)==1: run_sen('A1A2')
else:
    job=ijob.Job(1 if 'dnn' in clsuse or 'cnn' in clsuse else 10)
    for s in senuse:
        if os.path.exists('stop'): job.cleanup(); raise SystemExit()
        job.start('run_'+s,run_sen,(s,))
    for s in senuse: job.res('run_'+s)


# %bg _ip.magic('run -i foo.py')

raise SystemExit()


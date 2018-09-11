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
    print('svm start  '+fea+'_'+s)
    #if not 'C' in kwargs: kwargs['C']=1
    #if not 'tol' in kwargs: kwargs['tol']=0.1
    csvm=isvm.trn(ftrn,fea,**kwargs)
    prob=isvm.evlp(csvm,ftst,fea)
    print('svm finish '+fea+'_'+s)
    return prob

def hmmtrn(ftrn,ftst,fea,s,kwargs={}):
    print('hmm start  '+fea+'_'+s)
    chmm=ihmm.trn(flst=ftrn,fea=fea,its=[0],states=3)
    nld=ihmm.evlp(chmm,flst=ftst,fea=fea)
    if icfg.get('exp')=='triclass':
        print('hmm finish '+fea+'_'+s)
    else:
        res=np.array(chmm['cls'])[nld.argmin(axis=1)]
        icls.cmp(ftst,res,'hmm finish '+fea+'_'+s)
    return nld

def dnntrn(ftrn,ftst,fea,s,kwargs={}):
    print('dnn start  '+fea+'_'+s)
    cdnn=idnn.trn(ftrn,ftst,fea=fea,dmod=dmod,**kwargs)
    if cdnn is None:
        print('dnn failed '+fea+'_'+s)
        return []
    else:
        prob=idnn.evlp(cdnn,ftst,fea=fea)
        if icfg.get('exp')=='triclass':
            print('dnn finish '+fea+'_'+s)
        else:
            res=np.array(cdnn['cls'])[prob.argmax(axis=1)]
            icls.cmp(ftst,res,'dnn finish '+fea+'_'+s)
        return prob
cnntrn=dnntrn

def dnntest(s,fea='sfa',**kwargs):
    ftrns=flstexpandsen(ftrn,s,okpat)
    ftsts=flstexpandsen(ftst,s,okpat)
    fdb=ifdb.load(s)
    fealnk(ftrns+ftsts,fdb)
    ftrns=icls.equalcls(ftrns)
    cdnn=idnn.trn(ftrns,ftsts,fea=fea,dmod=dmod,**kwargs)
    if not cdnn is None: cdnn['idim']=ftrns[0][fea].shape
    return cdnn

import random
def dnnrnd():
    xip=int((random.random()**2)*1000)+10
    dp=(random.random()**5)*0.5
    kwargs={}
    kwargs['batch_size']=256
    kwargs['max_iter']=500
    kwargs['s']=sen[int(random.random()*len(sen))]
    kwargs['fea']=['sig','pfa','sfa'][int(random.random()*3)]
    kwargs['base_lr']=(random.random()**2)*0.1+0.001
    lay=[('ip',xip),('relu',),('ip',)]
    if True:
        o=int(random.random()*15)+10
        if kwargs['fea']=='sig':
            k=[int(random.random()*27)+3,1]
            s=[int(random.random()*max(1,min(15,k[0]-3)))+1,1]
        else:
            k=[int(random.random()*7)+3,
               int(random.random()*15)+3]
            s=[int(random.random()*max(1,min(5,k[0]-3)))+1,
               int(random.random()*max(1,min(9,k[0]-3)))+1]
        lay=[('conv',k,o,s)]+lay
    kwargs['lay']=lay
    # c 16,7,2,0,1 # ip 498
    #kwargs={'fea': 'pfa', 's': 'A1A2', 'max_iter': 200, 'weight_decay': 0.10661077392254943, 'lay': [('conv', 3, 18, 1), ('ip', 824), ('dropout', 0.00038962239039409444), ('relu',), ('ip',)], 'base_lr': 0.08700124692413123}
    stat=[]
    xpar=0
    for i in range(3):
        print(kwargs)
        r=dnntest(**kwargs)
        if r is None: break
        xpar=idnn._netstat(r,r['idim'],prt=False)
        if np.array(r['stat'])[:,2].max()<0.5: print("MAX-ACC<0.5"); break
        stat.append(np.array(r['stat']))
    print('/dnnrnd_'+str(int(time.time())))
    np.save(dlog+'/dnnrnd/dnnrnd_'+str(int(time.time()))+'_arg.npy',kwargs)
    np.save(dlog+'/dnnrnd/dnnrnd_'+str(int(time.time()))+'_par.npy',xpar)
    if len(stat)==3: np.save(dlog+'/dnnrnd/dnnrnd_'+str(int(time.time()))+'_stat.npy',stat)

def dnnrndloop():
    while not os.path.exists('stop'): dnnrnd()

if len(sys.argv)<2: raise ValueError("Usage: "+sys.argv[0]+" CFG [-n]")
icfg.Cfg(sys.argv[1])

ftrn=icfg.readflst('train')
ftst=icfg.readflst('test')
#fdev=[] if icfg.get('flist.dev') is None else icfg.readflst('dev')

dmod=icfg.getdir('model')
dlog=icfg.getdir('log')
sen=getsensors()

senuse=sen#[:1]

#clsuse=['hmm']
#clsuse=['svm']
#clsuse=['dnn']
#clsuse=['cnn','dnn']
clsuse=['cnnb','cnn','dnn']

#feause=['pfa']
feause=['pfa','sfa','sig']

okpat='Z0[0-2]' if icfg.get('db')=='izfp/cfk' else 'Z00'

maxjob=12

if len(sys.argv)>2 and sys.argv[2]=='-nn': raise SystemExit()

def run_sen(s):

    print("flst [%s]"%(s))
    ftrns=flstexpandsen(ftrn,s,okpat)
    ftsts=flstexpandsen(ftst,s,okpat)

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

    ftrns=icls.equalcls(ftrns)
    if len(sys.argv)>2 and sys.argv[2]=='-n': return

    for cls in clsuse:
        for fea in feause:
            if cls=='hmm' and ftrns[0][fea].shape[-1]>40: continue
            resfn=os.path.join(dlog,'res_'+cls+'_'+fea+'_'+s+'.npy')
            if os.path.exists(resfn): continue
            print('####################',fea,cls,s,'####################')
            kwargs=icfg.get('trnargs.%s.%s'%(cls,fea))
            if kwargs is None:
                if cls=='dnn' or cls[:3]=='cnn': continue
                kwargs={}
            else:
                print('trnargs = '+kwargs)
                kwargs=eval(kwargs)
            fnctrn=eval(cls[:3]+'trn')
            res=fnctrn(ftrns,ftsts,fea,s,kwargs)
            np.save(resfn,res)

if len(senuse)==1: run_sen(senuse[0])
else:
    job=ijob.Job(1 if len([cls for cls in clsuse if cls=='dnn' or cls[:3]=='cnn'])>0 else maxjob)
    for s in senuse:
        if os.path.exists('stop'): job.cleanup(); raise SystemExit()
        job.start('run_'+s,run_sen,(s,))
    for s in senuse: job.res('run_'+s)

# %bg _ip.magic('run -i foo.py')

raise SystemExit()


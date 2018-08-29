#!/usr/bin/python3

import os
import sys
import re
import time
import importlib
import numpy as np

import multiprocessing as mlp

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
importlib.reload(ipl)
importlib.reload(isig)
importlib.reload(ifea)
importlib.reload(icfg)
importlib.reload(isvm)
importlib.reload(ihmm)
importlib.reload(idnn)
importlib.reload(icls)
importlib.reload(idat)

def getsensors():
    fn=icfg.getfile('am.sensors','info','sensors.txt')
    if fn is None: return None
    fd=open(fn,'r')
    sen=[]
    for line in fd.readlines():
        line=re.sub('#.*|\n|\r|^[ \t]+|[ \t]+$','',line)
        if len(line)>0: sen.append(line)
    return sen

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

def job_start(name,fnc,args):
    def jobi(pipe,fnc,args): pipe.send(fnc(*args))
    ppipe,cpipe=mlp.Pipe()
    job_start.prc[name]={'hnd':mlp.Process(target=jobi,args=(cpipe,fnc,args)),'pipe':ppipe}
    if len(job_start.run)>=16:
        job_start.prc[job_start.run[0]]['hnd'].join()
        job_start.run=job_start.run[1:]
    job_start.prc[name]['hnd'].start()
    job_start.run.append(name)
job_start.prc={}
job_start.run=[]

def job_res(name):
    job_start.prc[name]['hnd'].join()
    return job_start.prc[name]['pipe'].recv()
                                         
if len(sys.argv)>2 and sys.argv[2]=='-nn': raise SystemExit()


if len(sys.argv)<2: raise ValueError("Usage: "+sys.argv[0]+" CFG [-n]")
icfg.Cfg(sys.argv[1])


dmod=icfg.getdir('model')
dlog=icfg.getdir('log')
dsig=icfg.getdir('sig')
sigext='.'+icfg.get('sig.ext','wav')
sen=getsensors()

if not 'fall' in locals():
    fall=icfg.readflst('all')
    falls={}
    for f in fall:
        for s in sen:
            fn=f['fn']+'.'+s
            falls[fn]={**f,'sen':s}
            falls[fn]['fn']=fn

print("fea")
t=time.time()
for f in falls.values():
    if t-time.time()>5:
        print("fea %i/%i"%(i,len(falls)))
        t=time.time()
    if not 'sig' in f:
        f['sig']=isig.load(os.path.join(dsig,f['fn']+sigext)).rmaxis()
        f['sig'].inc[0]=1/6250000
    if not 'pfa' in f:
         fea=ifea.fft(f['sig'],crate=icfg.get('pfa.crate'),wlen=icfg.get('pfa.wlen')).db()
         ifea.cavg(fea,rat=4,inplace=True)
         fea.sel(axis=len(fea.shape)-1,len=icfg.get('pfa.dim'),inplace=True)
         f['pfa']=fea

print("flst")
ftrn=icfg.readflst('train')
ftst=icfg.readflst('test')

ftrns={}
for strn in sen:
    ftrns[strn]=[]
    for f in ftrn:
        for s in sen:
            fn=f['fn']+'.'+s
            ftrns[strn].append({**falls[fn]})
            if strn!=s: ftrns[strn][-1]['lab']='Zxx'
    ftrns[strn]=icls.equalcls(ftrns[strn])
ftsts={}
for s in sen:
    ftsts[s]=[]
    for f in ftst:
        fn=f['fn']+'.'+s
        ftsts[s].append(falls[fn])

if len(sys.argv)>2 and sys.argv[2]=='-n': raise SystemExit()

for s in sen:
    if len([True for f in ftrns[s]+ftsts[s] if not 'sfa' in f])==0: continue
    print('sfa '+s)
    sfa=ifea.Sfa(ftrns[s],'pfa')
    #sfa.save(os.path.join(dmod,'sfa_'+s))
    for f in ftrns[s]+ftsts[s]:
        if not 'sfa' in f: f['sfa']=sfa.do(f['pfa'])

def svmtrn(ftrn,ftst,fea):
    print('svm start  '+s)
    csvm=isvm.trn(ftrn,fea)
    prob=isvm.evlp(csvm,ftst,fea)[:,0]
    print('svm finish '+s)
    return prob

fea='sig'
for s in sen: job_start(s,svmtrn,(ftrns[s],ftsts[s],fea))
prob=[job_res(s) for s in sen]
np.save(os.path.join(dlog,'psvm_'+fea+'.npy'),prob)

lab=rle([f['lab'] for f in ftsts[s]])
res=np.mean(prob,axis=0)
res=[(int(l[2][1:]),res[l[0]:l[0]+l[1]]) for l in lab]
res=[(l,np.mean(r),np.std(r)) for (l,r) in res]
#ipl.p2(res,err='y',xrange=(-1,40))
#ipl.cm(psvm)


# %bg _ip.magic('run -i foo.py')

raise SystemExit()

#dnn_arg={
#    'ffa': {'lay': [('conv',11,16,(6,4)),('pool',3,2),('ip',100),('relu',),('ip',)], 'max_iter': 500},
#    'ffd': {'lay': [('conv',11,16,(6,4)),('pool',3,2),('ip',100),('relu',),('ip',)], 'max_iter': 500},
#    'fft': {'base_lr': 0.1},
#    'pfa': {'lay': [('conv',11,16,(1,4)),('pool',3,2),('ip',100),('relu',),('ip',)], 'max_iter': 500},
#    'sfa': {'lay': [('conv',7,20),('pool',2,2),('ip',400),('relu',),('ip',)]},
#}
#for fea in ['ffa','ffd','fft','pfa','sfa']:
#    csvm=isvm.trn(ftrn,fea)
#    icls.cmp(ftst,isvm.evl(csvm,ftst,fea),name='svm_'+fea)
#    isvm.save(csvm,'model/svm_'+fea)
#    if idnn.caffe.found:
#        cdnn=idnn.trn(ftrn,ftst,fea,**dnn_arg[fea])
#        icls.cmp(ftst,idnn.evl(cdnn,ftst,fea),name='dnn_'+fea)
#        idnn.save(cdnn,'model/dnn_'+fea)
#        #ipl.p2(np.array(cdnn['stat']).transpose()[1:])
#
#
##chmm=ihmm.trn(ftrn,'sfa',[3,5,7,9])
#chmm=ihmm.load('model/hmm_sfa')
#icls.cmp(ftst,ihmm.evl(chmm,ftst,'sfa'),name='hmm_sfa')
##ihmm.save(chmm,'model/hmm_sfa')
#
##ipl.p2(np.array(idnn.trn(ftrn,ftst,'pfa',lay=[('ip',500),('relu',),('ip',)])['stat']).transpose()[1:])

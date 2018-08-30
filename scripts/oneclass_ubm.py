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

def feaana(fns):
    ret={}
    for fn in fns:
        ret[fn]=fx={}

        fx['sig']=isig.load(os.path.join(dsig,fn+sigext)).rmaxis()
        fx['sig'].inc[0]=1/6250000

        fea=ifea.fft(fx['sig'],crate=icfg.get('pfa.crate'),wlen=icfg.get('pfa.wlen')).db()
        ifea.cavg(fea,rat=4,inplace=True)
        fea.sel(axis=len(fea.shape)-1,len=icfg.get('pfa.dim'),inplace=True)
        fx['pfa']=fea

    return ret


def svmtrn(ftrn,ftst,fea,s):
    print('svm start  '+s)
    csvm=isvm.trn(ftrn,fea)
    prob=isvm.evlp(csvm,ftst,fea)[:,0]
    print('svm finish '+s)
    return prob

def hmmtrn(ftrn,ftst,fea,s):
    print('hmm start  '+s)
    ftrn0=[f for f in ftrn if f['lab']=='Z00']
    chmm=ihmm.trn(flst=ftrn0,fea=fea,its=[0])
    prob=ihmm.evlp(chmm,flst=ftst,fea=fea)[:,0]
    prob/=[f[fea].shape[0] for f in ftst]
    prob*=-1
    print('hmm finish '+s)
    return prob

def dnntrn(ftrn,ftst,fea,s,**kwargs):
    print('dnn start  '+s)
    cdnn=idnn.trn(ftrn,ftst,fea=fea,dmod=dmod,**kwargs)
    prob=idnn.evlp(cdnn,ftst,fea=fea)[:,0]
    print('dnn finish '+s)
    return prob

def dnntest(s,fea='sfa',**kwargs):
    ftst=ftsts[s]
    ftrn=ftrns[s]
    ftst2=[{**f} for f in ftst]
    for f in ftst2:
        if f['lab']!='Z00': f['lab']='Zxx'
    cdnn=idnn.trn(ftrn,ftst2,fea=fea,dmod=dmod,**kwargs)
    res=idnn.evl(cdnn,ftst,fea=fea)
    lab=[f['lab'] for f in ftst]
    c=len([True for l,r in zip(lab,res) if l==r])
    print(c,len(lab),c/len(lab))

if len(sys.argv)>2 and sys.argv[2]=='-nn': raise SystemExit()

if len(sys.argv)<2: raise ValueError("Usage: "+sys.argv[0]+" CFG [-n]")
icfg.Cfg(sys.argv[1])

print("flst")
ftrn=icfg.readflst('train')
ftst=icfg.readflst('test')

dmod=icfg.getdir('model')
dlog=icfg.getdir('log')
dsig=icfg.getdir('sig')
sigext='.'+icfg.get('sig.ext','wav')
sen=getsensors()

ftrns={}
for strn in sen:
    ftrns[strn]=[]
    for f in ftrn:
        for s in sen:
            fn={**f}
            fn['fn']+='.'+s
            if strn!=s: fn['lab']='Zxx'
            ftrns[strn].append(fn)
    ftrns[strn]=icls.equalcls(ftrns[strn])
ftsts={}
for s in sen:
    ftsts[s]=[]
    for f in ftst:
        fn={**f}
        fn['fn']+='.'+s
        ftsts[s].append(fn)

if not 'fdb' in locals(): fdb={}

print("fea")
feado=[f['fn'] for flst in [*ftsts.values(),*ftrns.values()] for f in flst if not f['fn'] in fdb]
feado=list({*feado})

t=time.time()
for i in range(len(feado)):
    if time.time()-t>5:
        print("fea %i/%i"%(i,len(feado)))
        t=time.time()
    fn=feado[i]
    fdb[fn]=feaana([fn])[fn]

#for i in range(0,len(feado),1000):
#    print('fea start  %i/%i'%(i,len(feado)))
#    ijob.thr_start('fea_%i'%(i),feaana,(feado[i:i+1000],))
#for i in range(0,len(feado),1000):
#    for fn,val in ijob.thr_res('fea_%i'%(i)).items(): fdb[fn]=val
#    print('fea finish %i/%i'%(i,len(feado)))

print("fealnk")
for flst in [*ftsts.values(),*ftrns.values()]:
    for f in flst:
        if not f['fn'] in fdb: raise ValueError('fea missing for: '+fn)
        for fea,val in fdb[f['fn']].items(): f[fea]=val


if len(sys.argv)>2 and sys.argv[2]=='-n': raise SystemExit()

for s in sen:
    if len([True for f in ftrns[s]+ftsts[s] if not 'sfa' in f])==0: continue
    print('sfa '+s)
    sfa=ifea.Sfa(ftrns[s],'pfa')
    #sfa.save(os.path.join(dmod,'sfa_'+s))
    for f in ftrns[s]+ftsts[s]:
        if not 'sfa' in f: f['sfa']=sfa.do(f['pfa'])

#fea='pfa'
#for s in sen: ijob.job_start('svmtrn_'+s,svmtrn,(ftrns[s],ftsts[s],fea,s))
#prob=[ijob.job_res('svmtrn_'+s) for s in sen]
#np.save(os.path.join(dlog,'prob_svm_'+fea+'.npy'),prob)

fea='pfa'
cls='dnn'
ijob.job_start.maxrun=1
fnctrn=eval(cls+'trn')
for s in sen: ijob.job_start(cls+'trn_'+s,fnctrn,(ftrns[s],ftsts[s],fea,s))
prob=[ijob.job_res(cls+'trn_'+s) for s in sen]
np.save(os.path.join(dlog,'prob_'+cls+'_'+fea+'.npy'),prob)


# %bg _ip.magic('run -i foo.py')

raise SystemExit()

#dnn_arg={
#    'ffa': {'lay': [('conv',11,16,(6,4)),('pool',3,2),('ip',100),('relu',),('ip',)], 'max_iter': 500},
#    'ffd': {'lay': [('conv',11,16,(6,4)),('pool',3,2),('ip',100),('relu',),('ip',)], 'max_iter': 500},
#    'fft': {'base_lr': 0.1},
#    'pfa': {'lay': [('conv',11,16,(1,4)),('pool',3,2),('ip',100),('relu',),('ip',)], 'max_iter': 500},
#    'sfa': {'lay': [('conv',7,20),('pool',2,2),('ip',400),('relu',),('ip',)]},
#}
#        cdnn=idnn.trn(ftrn,ftst,fea,**dnn_arg[fea])

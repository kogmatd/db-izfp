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

def sigget(fns):
    for f in fns:
        sig=isig.load(os.path.join(dsig,f['fn']+sigext)).rmaxis()
        sig.inc[0]=1/icfg.get('sig.srate')
        f['sig']=sig

def pfaget(fns):
    for f in fns:
        if not 'sig' in f: raise ValueError("feaget without sig for: "+f['fn'])
        fea=ifea.fft(f['sig'],crate=icfg.get('pfa.crate'),wlen=icfg.get('pfa.wlen')).db()
        ifea.cavg(fea,rat=4,inplace=True)
        fea.sel(axis=len(fea.shape)-1,len=icfg.get('pfa.dim'),inplace=True)
        f['pfa']=fea

def sfaget(ftrn,ftst):
    sfa=ifea.Sfa(ftrn,'pfa')
    #sfa.save(os.path.join(dmod,'sfa_'+s))
    for f in ftrn+ftst:
        if not 'sfa' in f: f['sfa']=sfa.do(f['pfa'])

def svmtrn(ftrn,ftst,fea,s,**kwargs):
    print('svm start  '+s)
    csvm=isvm.trn(ftrn,fea,**kwargs)
    prob=isvm.evlp(csvm,ftst,fea)[:,0]
    print('svm finish '+s)
    return prob

def hmmtrn(ftrn,ftst,fea,s):
    def _softmax(x,f,f2):
        x/=np.array(f).reshape((-1,1))*f2
        x=np.exp(x-x.max())
        x/=x.sum(1).reshape((-1,1))
        return x
    print('hmm start  '+s)
    chmm=ihmm.trn(flst=ftrn,fea=fea,its=[0])
    nld=ihmm.evlp(chmm,flst=ftst,fea=fea)
    prob=_softmax(nld,[f[fea].shape[0] for f in ftst],-40)
    prob=prob.take(0,-1)
    print('hmm finish '+s)
    return np.concatenate((prob.reshape(-1,1),nld),axis=1)
    #return prob

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
    kwargs['base_lr']=(random.random()**2)*0.1+0.02
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

if len(sys.argv)>2 and sys.argv[2]=='-nn': raise SystemExit()

if len(sys.argv)<2: raise ValueError("Usage: "+sys.argv[0]+" CFG [-n]")
icfg.Cfg(sys.argv[1])

print("flst")
ftrn=icfg.readflst('train')
ftst=icfg.readflst('test')
#fdev=[] if icfg.get('flist.dev') is None else icfg.readflst('dev')

dmod=icfg.getdir('model')
dlog=icfg.getdir('log')
dsig=icfg.getdir('sig')
sigext='.'+icfg.get('sig.ext','wav')
sen=getsensors()
senuse=sen#[:1]

ftrns={}
for strn in senuse:
    ftrns[strn]=[]
    for f in ftrn:
        for s in sen:
            fn={**f}
            fn['fn']+='.'+s
            if strn!=s: fn['lab']='Zxx'
            else: fn['lab']='Z00'
            ftrns[strn].append(fn)
    ftrns[strn]=icls.equalcls(ftrns[strn])
ftsts={}
for s in senuse:
    ftsts[s]=[]
    for f in ftst:
        fn={**f}
        fn['fn']+='.'+s
        ftsts[s].append(fn)

if not 'fdb' in locals(): fdb=ifdb.load()

for flst in [*ftsts.values(),*ftrns.values()]:
    for f in flst:
        if not f['fn'] in fdb: fdb[f['fn']]={'fn':f['fn']}

thr=ijob.Thr(16)
fdb_chg=False
for typ in ['sig','pfa']:
    print(typ)
    do=[f for f in fdb.values() if not typ in f]
    if len(do)==0: continue
    fdb_chg=True
    fnc=eval(typ+'get')

#    t=time.time()
#    for i in range(len(do)):
#        if time.time()-t>5:
#            print("%s %i/%i"%(typ,i,len(do)))
#            t=time.time()
#        fnc([do[i]])

    for i in range(0,len(do),1000):
        if os.path.exists('stop'): thr.cleanup(); raise SystemExit()
        print('%s start  %i/%i'%(typ,i,len(do)))
        thr.start('%s_%i'%(typ,i),fnc,(do[i:min(i+1000,len(do))],))
    for i in range(0,len(do),1000): thr.res('%s_%i'%(typ,i))

if senuse==sen and fdb_chg: ifdb.save(fdb)

print("fealnk")
for flst in [*ftsts.values(),*ftrns.values()]:
    for f in flst:
        if not f['fn'] in fdb: raise ValueError('fea missing for: '+fn)
        for fea,val in fdb[f['fn']].items(): f[fea]=val


print('sfa')
do=[s for s in senuse if len([True for f in ftrns[s]+ftsts[s] if not 'sfa' in f])!=0]
for s in do:
    if os.path.exists('stop'): thr.cleanup(); raise SystemExit()
    print('sfa start  '+s)
    thr.start('sfa_'+s,sfaget,(ftrns[s],ftsts[s]))
for s in do: thr.res('sfa_'+s)

if len(sys.argv)>2 and sys.argv[2]=='-n': raise SystemExit()
#dnnrndloop()
#raise SystemExit()

for cls in ['svm','hmm']:
    feas=['pfa','sfa']
    if cls!='hmm': feas.append('sig')
    for fea in feas:
        probfn=os.path.join(dlog,'prob_'+cls+'_'+fea+'.npy')
        if os.path.exists(probfn): continue
        job=ijob.Job(22 if cls!='dnn' else 1)
        fnctrn=eval(cls+'trn')
        for s in sen:
            if os.path.exists('stop'): job.cleanup(); raise SystemExit()
            job.start(cls+'trn_'+s,fnctrn,(ftrns[s],ftsts[s],fea,s))
        prob=[job.res(cls+'trn_'+s) for s in sen]
        np.save(probfn,prob)

#x=dnntest('A1A2','sig',max_iter=200,lay=[('conv',[7,1],20,[5,1]),('ip',200),('dropout',0.5),('relu',),('ip',)])
#x=dnntest('A1A2','pfa',max_iter=200,lay=[('ip',20),('relu',),('ip',)],weight_decay=0.1)
#x=dnntest('A1A2','pfa',max_iter=200,lay=[('ip',200),('dropout',0.5),('relu',),('ip',)])
#x=dnntest('A1A2','pfa',max_iter=200,lay=[('conv',7,20,5),('ip',200),('dropout',0.5),('relu',),('ip',)],weight_decay=0.01)

# %bg _ip.magic('run -i foo.py')

raise SystemExit()


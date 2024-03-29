import numpy as np
import os
import re

import icfg
import isig
import ifea
import icls
import ipl
import ihlp

def argv2resfns(pat,fns):
    dlog=icfg.getdir('log')
    fns=[*fns]
    if len(fns)==0: fns=[dlog]
    i=0
    while i<len(fns):
        if not os.path.exists(fns[i]): fns[i]=os.path.join(dlog,fns[i])
        if os.path.isdir(fns[i]):
            ins=[os.path.join(fns[i],f) for f in os.listdir(fns[i]) if f[:len(pat)]==pat and f[-4:]=='.npy']
            fns=fns[:i]+ins+fns[i+1:]
            i+=len(ins)
        else: i+=1
    return fns
    

def getsensors():
    fn=icfg.getfile('am.sensors','info','sensors.txt')
    if fn is None: return None
    fd=open(fn,'r')
    sen=[]
    for line in fd.readlines():
        line=re.sub('#.*|\n|\r|^[ \t]+|[ \t]+$','',line)
        if len(line)>0: sen.append(line)
    return sen

def sigget(f):
    dsig=icfg.getdir('sig')
    sigext='.'+icfg.get('sig.ext','wav')
    sig=isig.load(os.path.join(dsig,f['fn']+sigext)).rmaxis()
    sig.inc=[1/icfg.get('sig.srate')]
    return sig

def pfaget(f):
    if not 'sig' in f: raise ValueError("feaget without sig for: "+f['fn'])
    fea=ifea.fft(f['sig'],crate=icfg.get('pfa.crate'),wlen=icfg.get('pfa.wlen')).db()
    ifea.cavg(fea,rat=4,inplace=True)
    fea.sel(axis=len(fea.shape)-1,len=icfg.get('pfa.dim'),inplace=True)
    return fea

def sfaget(ftrn,ftst,fdb):
    if any(not 'sfa' in f for f in ftrn+ftst):
        sfa=ifea.Sfa(ftrn,'pfa')
        fdb.analyse('sfa',lambda f:sfa.do(f['pfa']),flst=ftrn+ftst,force=True)

def plt_clsmeanstd(res,flst,**kwargs):
    icls.labf(flst)
    lab=ihlp.rle([f['labf'] for f in flst])
    reslab=[(l[2],res.take(range(l[0],l[0]+l[1]),axis=-1)) for l in lab]
    reslab=[(l,r.reshape(-1,r.shape[-1])) for (l,r) in reslab]
    resmean=[[(l,np.mean(r[i]),np.std(r[i])) for (l,r) in reslab] for i in range(reslab[0][1].shape[0])]
    resmean=np.array(resmean)
    ipl.p2(resmean,err='y',xrange=(-1,resmean[0][-1][0]+1),**kwargs)

#!/usr/bin/python3

import os
import sys
import re
import time
import importlib
import numpy as np

sys.path.append(os.environ['UASR_HOME']+'-py')

import ipl
import isig
import ifea
import icfg
import isvm
#import imodsvm
import ihmm
import idnn
import iktf
import icls
import idat
import ijob
import ifdb
import ihelp
from ihelp import *
icnn=idnn

def svmtrn(ftrn,ftst,fea,s,kwargs={}):
    print('svm start  '+fea+'_'+s)
    csvm=isvm.trn(ftrn, fea, **kwargs)
    restrn=isvm.evl(csvm, ftrn, fea)
    restst=isvm.evl(csvm, ftst, fea)
    if not regression:
        icls.cmp(ftrn, restrn, 'svm finish ')
        icls.cmp(ftst, restst, 'svm finish ')
    print('svm finish '+fea+'_'+s)
    return (csvm,restrn,restst)

#def svmtrn(ftrn,ftst,fea,s,kwargs={}):
#    print('svm start  '+fea+'_'+s)
#    svmcls=imodsvm.ModSVM()
#    svmcls.trn(ftrn,fea)
#    svmcls.evl(ftrn,fea)
#    svmcls.evl(ftst,fea)
#    print('svm finish '+fea+'_'+s)
#    return None

def hmmtrn(ftrn,ftst,fea,s,kwargs={}):
    print('hmm start  '+fea+'_'+s)
    chmm=ihmm.trn(flst=ftrn,fea=fea,its=[0],states=3,**kwargs)
    nldtrn=ihmm.evlp(chmm,flst=ftrn,fea=fea)
    nldtst=ihmm.evlp(chmm,flst=ftst,fea=fea)
    if icfg.get('exp')=='triclass' or icfg.get('trn.regression')==True:
        print('hmm finish '+fea+'_'+s)
    else:
        restrn=np.array(chmm['cls'])[nldtrn.argmin(axis=1)]
        icls.cmp(ftrn,restrn,'hmm finish '+fea+'_'+s)
        restst = np.array(chmm['cls'])[nldtst.argmin(axis=1)]
        icls.cmp(ftst, restst, 'hmm finish ' + fea + '_' + s)
    return (chmm,nldtrn,nldtst)

def ktftrn(ftrn,ftst,fea,s,kwargs={}):
    print('dnn start  '+s)
    config = dict()
    config['batchsize'] = 100
    config['lay'] = [('relu',600),('batch',), ('dropout',0.5), ('relu',300),('batch',),('dropout',0.5)]
    ktf = iktf.ModKeras()
    ktf.trn(ftrn, fea)
    restrn=ktf.evl(ftrn, fea, prob=True)
    restst=ktf.evl(ftst, fea, prob=True)
    print('dnn stop  ' + s)
    return (ktf.mod,restrn,restst)

def dnntrn(ftrn,ftst,fea,s,kwargs={}):
    print('dnn start  '+fea+'_'+s)
    cdnn=idnn.trn(ftrn,ftst,fea=fea,dmod=dmod,**kwargs)
    if cdnn is None:
        print('dnn failed '+fea+'_'+s)
        return (None,None,[])
    else:
        restrn=idnn.evlp(cdnn,ftrn,fea=fea)
        restst=idnn.evlp(cdnn,ftst,fea=fea)
        if icfg.get('exp')=='triclass' or icfg.get('trn.regression')==True:
            print('dnn finish '+fea+'_'+s)
        else:
            res=np.array(cdnn['cls'])[prob.argmax(axis=1)]
            icls.cmp(ftst,res,'dnn finish '+fea+'_'+s)
        return (cdnn,restrn,restst)
cnntrn=dnntrn

def dnntest(s,fea='sfa',**kwargs):
    fdb=ifdb.load(s)
    ftrns=ftrn.expandsensor(s,fdb).maplab(labmap)
    ftsts=ftst.expandsensor(s,fdb).maplab(labmap)
    if icfg.get('trn.regression')!=True: ftrns=ftrns.equalcls()
    cdnn=idnn.trn(ftrns,ftsts,fea=fea,dmod=dmod,verbose=True,**kwargs)
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
icfg.Cfg(*sys.argv[1:])

ftrn=icfg.readflst('train')
ftst=icfg.readflst('test')

dmod=icfg.getdir('model')
dlog=icfg.getdir('log')
sen=getsensors()
regression=icfg.get('trn.regression')==True

senuse=sen
feause=['pfa','sfa','sig']
clsuse=['hmm','svm','cnnb','cnn','dnn','keras']
if not icfg.get('senuse') is None: senuse=icfg.get('senuse').split(',')
if not icfg.get('feause') is None: feause=icfg.get('feause').split(',')
if not icfg.get('clsuse') is None: clsuse=icfg.get('clsuse').split(',')

labmap={}
if icfg.get('db')=='izfp/cfk': labmap={'Z0[0-2]':'Z00'}

maxjob=18

if '-nn' in sys.argv: raise SystemExit()

def run_sen(s):

    print("flst [%s]"%(s))
    fdb=ifdb.Fdb(s)
    ftrns=ftrn.expandsensor(s,fdb).maplab(labmap)
    ftsts=ftst.expandsensor(s,fdb).maplab(labmap)
    for typ in ['sig','pfa']: fdb.analyse(typ,eval(typ+'get'))
    sfaget(ftrns,ftsts,fdb)
    fdb.save()
    if not regression: ftrns=ftrns.equalcls()
    if '-n' in sys.argv: return

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
            kwargs['regression']=regression
            fnctrn=eval(cls[:3]+'trn')
            for i in range(3):
                (mod,restrn,restst)=fnctrn(ftrns,ftsts,fea,s,kwargs)
                if len(restst)>0:
                    np.save(resfn,restst)
                    np.save(resfn[:-4]+'_trn.npy',restrn)
                    eval('i'+cls[:3]+'.save')(mod,resfn[:-4]+'.model')
                    break

if len(senuse)==1: run_sen(senuse[0])
else:
    job=ijob.Job(1 if len([cls for cls in clsuse if cls=='dnn' or cls[:3]=='cnn'])>0 else maxjob)
    for s in senuse:
        if os.path.exists('stop'): job.cleanup(); raise SystemExit()
        job.start('run_'+s,run_sen,(s,))
    for s in senuse: job.res('run_'+s)

# %bg _ip.magic('run -i foo.py')

raise SystemExit()


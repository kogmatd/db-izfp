#!/usr/bin/python3

import numpy as np
import os
import sys
import importlib

sys.path.append(os.path.join(os.environ['HOME'],'audiomix/anawav'))

import ipl
import icfg
importlib.reload(ipl)
importlib.reload(icfg)

if len(sys.argv)<2: raise ValueError("Usage: "+sys.argv[0]+" CFG {PAR}")
icfg.Cfg(sys.argv[1])

dlog=icfg.getdir('log')
dn=os.path.join(dlog,'dnnrnd')

res=[]
for fn in os.listdir(dn):
    if fn[-8:]!='_arg.npy': continue
    fna=os.path.join(dn,fn)
    fns=fna[:-8]+'_stat.npy'
    if not os.path.exists(fns): continue
    res.append({'fna':fna,'fns':fns})
    fnx=fna[:-8]+'_stat_allsen.npy'
    if os.path.exists(fnx): res[-1]['fnx']=fnx

def arg2par(arg):
    par={}
    par['weight_decay']=arg['weight_decay']
    par['base_lr']=arg['base_lr']
    lay=arg['lay']
    par['conv']=1 if lay[0][0]=='conv' else 0
    par['conv_k']=lay[0][1] if lay[0][0]=='conv' else 0
    par['conv_o']=lay[0][2] if lay[0][0]=='conv' else 0
    par['conv_s']=lay[0][3] if lay[0][0]=='conv' else 0
    if type(par['conv_k']) is list: par['conv_k']=max(par['conv_k'])
    if type(par['conv_s']) is list: par['conv_s']=max(par['conv_s'])
    par['ip']=lay[par['conv']+0][1]
    par['dropout']=lay[par['conv']+1][1]
    return par

for r in res:
    r['arg']=eval(str(np.load(r['fna'])))
    r['stat']=np.load(r['fns'])
    r['best']=np.max(np.min(np.array(r['stat'])[:,:,2],axis=0))
    r['par']=arg2par(r['arg'])
    if 'fnx' in r:
        r['stat_allsen']=np.load(r['fnx'])
        r['best_allsen']=np.max(np.min(np.array(r['stat'])[:,:,2],axis=0))

print('min-best>0.8: %.2f'%(np.min([r['best'] for r in res if r['best']>0.8])))

for key in ['s','fea']:
    for v in sorted({r['arg'][key] for r in res}):
        x=[r['best'] for r in res if r['arg'][key]==v]
        c=np.sum(np.array(x)>0.8)
        ax=[r['best_allsen'] for r in res if r['arg'][key]==v and 'best_allsen' in r]
        ac=np.sum(np.array(ax)>0.8)
        print('%s: %4i/%-4i - %.2f%s'%(v,c,len(x),c/len(x),' [allsen:%i/%i]'%(ac,len(ax)) if key=='fea' else ''))

for fea in['sig','pfa','sfa']:
    for p in sys.argv[2:]:
        a=[r['par'][p] for r in res if r['arg']['fea']==fea and r['best']>0.8]
        b=[r['par'][p] for r in res if r['arg']['fea']==fea and r['best']<0.8]
        bins=np.linspace(0,1,30)
        if p=='base_lr': bins=(bins**2)*0.1+0.001
        if p=='weight_decay': bins=(bins**4)*0.4
        if p=='conv_k': bins=bins*5+3
        if p=='conv_o': bins=bins*15+10
        if p=='conv_s': bins=bins*3+1
        if p=='ip': bins=(bins**2)*1000+10
        if p=='dropout': bins=(bins**5)*0.5
        ha=np.histogram(a,bins)
        hb=np.histogram(b,bins)
        hx=np.mean([ha[1][1:],ha[1][:-1]],axis=0)
        #ipl.p2([hx,[ha[0],hb[0]]],title=fea+' '+p)
        ipl.p2([hx,ha[0]/(ha[0]+hb[0]+(ha[0]+hb[0]==0))*(ha[0]+hb[0]!=0)],title=fea+' '+p)

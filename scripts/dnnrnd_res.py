#!/usr/bin/python3

import numpy as np
import os
import sys
import importlib

sys.path.append(os.path.join(os.environ['HOME'],'audiomix/anawav'))

import ipl
import icfg
import idnn
importlib.reload(ipl)
importlib.reload(icfg)
importlib.reload(idnn)

if len(sys.argv)<2: raise ValueError("Usage: "+sys.argv[0]+" CFG {PAR}")
icfg.Cfg(sys.argv[1])

dlog=icfg.getdir('log')
dn=os.path.join(dlog,'dnnrnd')

res=[]
for fn in os.listdir(dn):
    if fn[-8:]!='_arg.npy': continue
    fna=os.path.join(dn,fn)
    res.append({'fna':fna})
    fns=fna[:-8]+'_stat.npy'
    if os.path.exists(fns): res[-1]['fns']=fns
    fnx=fna[:-8]+'_stat_allsen.npy'
    if os.path.exists(fnx): res[-1]['fnx']=fnx
    res[-1]['fnp']=fna[:-8]+'_par.npy'

def arg2par(arg):
    par={}
    if 'weight_decay' in arg: par['weight_decay']=arg['weight_decay']
    par['base_lr']=arg['base_lr']
    lay=arg['lay']
    par['conv']=1 if lay[0][0]=='conv' else 0
    par['conv_k']=lay[0][1] if lay[0][0]=='conv' else 0
    par['conv_o']=lay[0][2] if lay[0][0]=='conv' else 0
    par['conv_s']=lay[0][3] if lay[0][0]=='conv' else 0
    if type(par['conv_k']) is list: par['conv_k']=max(par['conv_k'])
    if type(par['conv_s']) is list: par['conv_s']=max(par['conv_s'])
    par['ip']=lay[par['conv']+0][1]
    if lay[par['conv']+1][0]=='dropout': par['dropout']=lay[par['conv']+1][1]
    return par

def lay2xpar(fea,lay):
    #if fea=='sig': idim=(1,2500)
    #elif fea=='pfa': idim=(24,47)
    #elif fea=='sfa': idim=(16,47)
    if fea=='sig': idim=(1,4352)
    elif fea=='pfa': idim=(105,32)
    elif fea=='sfa': idim=(105,16)
    odim=2
    fnet=os.path.join(dlog,'auxnet.prototxt')
    lay=[*lay[:-1],(lay[-1][0],odim,*lay[-1][1:])]
    idnn._netgen(fdb=None,fnet=fnet,lay=lay,batch_size=1,idim=idim)
    return idnn._netstat({'netspec':fnet},idim,prt=False)

for r in res:
    r['arg']=eval(str(np.load(r['fna'])))
    if 'fns' in r:
        r['stat']=np.load(r['fns'])
        r['best']=np.max(np.min(np.array(r['stat'])[:,:,2],axis=0))
    else: r['best']=0
    r['par']=arg2par(r['arg'])
    if os.path.exists(r['fnp']): r['par']['xpar']=int(np.load(r['fnp']))
    else:
        r['par']['xpar']=lay2xpar(r['arg']['fea'],r['arg']['lay'])
        np.save(r['fnp'],r['par']['xpar'])
    if 'fnx' in r:
        r['stat_allsen']=np.load(r['fnx'])
        r['best_allsen']=np.max(np.min(np.array(r['stat_allsen'])[:,:,2],axis=0))

thrbest=0.7
#print('min-best>%.1f: %.2f'%(thrbest,np.min([r['best'] for r in res if r['best']>thrbest])))

for key in ['fea']: #['s','fea']:
    for v in sorted({r['arg'][key] for r in res}):
        x=[r['best'] for r in res if r['arg'][key]==v]
        c=np.sum(np.array(x)>thrbest)
        ax=[r['best_allsen'] for r in res if r['arg'][key]==v and 'best_allsen' in r]
        ac=np.sum(np.array(ax)>thrbest)
        print('%s: %4i/%-4i - %.2f%s'%(v,c,len(x),c/len(x),' [allsen:%i/%i]'%(ac,len(ax)) if key=='fea' else ''))

def cor(v):
    a=np.array(v)[:,0]
    b=np.array(v)[:,1]
    a=a-np.mean(a)
    b=b-np.mean(b)
    if np.max(np.abs(a))==0: return 0
    if np.max(np.abs(b))==0: return 0
    return np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))

for fea in['sig','pfa','sfa']:
    for p in par:
        c=cor([(r['best'],r['par'][p]) for r in res if r['arg']['fea']==fea])
        if c<0.3: continue
        print("cor %s: %s %s => %.2f"%(fea,'best',p,c))
        

par=list(res[0]['par'].keys())
par.remove('xpar')
par.remove('conv')
for fea in['sig','pfa','sfa']:
    for ip1 in range(len(par)):
        for ip2 in range(ip1+1,len(par)):
            p1=par[ip1]
            p2=par[ip2]
            v=[[r['par'][p1],r['par'][p2]] for r in res if r['arg']['fea']==fea and r['best']>thrbest]
            if len(v)==0: continue
            c=cor(v)
            if c<0.3: continue
            print("cor %s: %s %s => %.2f"%(fea,p1,p2,c))

#for fea in['sig','pfa','sfa']:
#    x=[r for r in res if 'best_allsen' in r and r['best_allsen']>0.8 and r['arg']['fea']==fea]
#    x=x[np.argmin([r['par']['xpar'] for r in x])]
#    print('##### best&smallest model for '+fea+' #####')
#    print(x['par'])
#    print(x['arg'])

for fea in['sig','pfa','sfa']:
    for p in sys.argv[2:]:
        a=[r['par'][p] for r in res if r['arg']['fea']==fea and r['best']>0.8]
        b=[r['par'][p] for r in res if r['arg']['fea']==fea and r['best']<0.8]
        bins=np.linspace(0,1,30)
        if p=='base_lr': bins=(bins**2)*0.1+0.001
        if p=='weight_decay': bins=(bins**4)*0.4
        if p=='conv_k': bins=bins*29+1
        if p=='conv_o': bins=bins*15+10
        if p=='conv_s': bins=bins*15+1
        if p=='ip': bins=(bins**2)*1000+10
        if p=='dropout': bins=(bins**5)*0.5
        if p=='xpar': bins*=np.max([r['par']['xpar'] for r in res])
        ha=np.histogram(a,bins)
        hb=np.histogram(b,bins)
        hx=np.mean([ha[1][1:],ha[1][:-1]],axis=0)
        ipl.p2([hx,[ha[0],hb[0]]],title=fea+' '+p)
        #ipl.p2([hx,ha[0]/(ha[0]+hb[0]+(ha[0]+hb[0]==0))*(ha[0]+hb[0]!=0)],title=fea+' '+p)


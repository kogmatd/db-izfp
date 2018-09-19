import subprocess
import os
import sys
import wave
import numpy as np

sys.path.append(os.environ['UASR_HOME']+'-py')

import ipl

if len(sys.argv)!=2:
    print("Usage: "+sys.argv[0]+" als|cfk")
    raise SystemExit()

db=sys.argv[1]
dbdir=os.environ['UASR_HOME']+'-data/izfp/'+db

flst=subprocess.check_output(['find',dbdir+'/common/ori/unzip','-name','*.dat']).decode('utf-8').split('\n')
flst.pop()

#flst=flst[:10]

def fn2wfn(fn):
    if fn[-4:]!='.dat': raise ValueError("file name error")
    fl=fn[:-4].split('/')
    fl[-2]+='_wav'
    wfn1=str.join('/',fl)+'1.wav'
    wfn2=str.join('/',fl)+'2.wav'
    return (wfn1,wfn2)

def wavload(fn):
    wd=wave.open(fn)
    if wd.getnchannels()!=1: raise ValueError("wav format error")
    if wd.getsampwidth()!=2: raise ValueError("wav format error")
    return np.frombuffer(wd.readframes(wd.getnframes()),np.int16)

def datload(fn):
    dat=np.fromfile(fn,'>f4')
    l=(len(dat)-64)//2
    return (dat[64:64+l],dat[64+l:])

def dcmp(fn,d,w):
    if d.shape!=w.shape: print('Shape missmatch in '+fn)
    else:
        x=d-w
        if np.max(x)-np.min(x)>1: print('Offset not constant (%i,%i) in %s'%(np.min(x),np.max(x),fn))
        dcmp.agg.append([np.mean(d),np.mean(x),np.mean(w)])
dcmp.agg=[]

for fn in flst:
    wfn1,wfn2=fn2wfn(fn)
    d1,d2=datload(fn)
    w1=wavload(wfn1)
    w2=wavload(wfn2)
    dcmp(fn,d1,w1)
    dcmp(fn,d2,w2)

ipl.p2(da.take([0,2],axis=1),style='.')

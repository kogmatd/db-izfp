import subprocess
import os
import sys

if len(sys.argv)<2:
    print("Usage: "+sys.argv[0]+" als|cfk [-ln]")
    raise SystemExit()

db=sys.argv[1]
dbdir=os.environ['UASR_HOME']+'-data/izfp/'+db
uzdir=dbdir+'/common/ori/unzip'

clsmap_als={
    'R0':'Z00',
    'R1':'Z01',
    'R2':'Z02',
    'R3':'Z03',
    'R4':'Z04',
    'R5':'Z05',
    'R6':'Z06',
    'R7':'Z07',
    'R8':'Z08',
    'R9':'Z09',
    'RA':'Z10',
    'RB':'Z11',
    'RC':'Z12',
    'RD':'Z13',
    'RE':'Z14',
    'RF':'Z15',
    'RG':'Z16',
    'RH':'Z17',
    'RI':'Z18',
    'RJ':'Z19',
    'RK':'Z20',
    'RL':'Z21',
    'RM':'Z22',
    'RN':'Z23',
    'RO':'Z24',
    'RP':'Z25',
    'RQ':'Z26',
    'RR':'Z27',
    'RS':'Z28',
    'RT':'Z29',
    'RU':'Z30',
    'RV':'Z31',
    'RW':'Z32',
    'RX':'Z33',
    'RY':'Z34',
    'RZ':'Z35',
    'R-':'Z36',
    'R+':'Z37',
}

clsmap_cfk={
    'Z0':'Z00',
    'Z1':'Z01',
    'Z2':'Z02',
    'Z3':'Z03',
    'Z4':'Z04',
    'Z5':'Z05',
    'Z6':'Z06',
    'Z7':'Z07',
}

excmap_cfk={
    'R100':'X1',
    'R350':'X2',
    'S600':'X3',
}

flst=subprocess.check_output(['find',uzdir,'-type','f']).decode('utf-8').split('\n')
flst.pop()

flst=[(*fn.split('/')[-2:],fn) for fn in flst]

fns={}

for dn,fn,fno in flst:
    if dn[-4:]=='_wav': dn=dn[:-4]
    if fn[:len(dn)+1]!=dn+'_': raise ValueError('dir and file name missmatch: %s <=> %s'%(dn,fn))
    if db=='als':
        if len(dn)!=6: raise ValueError('parse error at dir name: '+dn)
        if fn[12]!='_': raise ValueError('parse error at file name: '+fn)
        cls1=fn[0:2]
        sen1=fn[2:4]
        cls2=fn[4:6]
        if cls2!='R1': raise ValueError('class missmatch: '+cls1+' <=> '+cls2)
        idx=fn[7:12]
        exc=''
        if not cls1 in clsmap_als: raise ValueError('class unkown: '+cls1)
        cls1=clsmap_als[cls1]

        if fn[-4:]=='.wav':
            if len(fn)!=19: raise ValueError('parse error at file name: '+fn)
            sen2=fn[13:15]
            seni=['']
        else:
            if len(fn)!=18: raise ValueError('parse error at file name: '+fn)
            if fn[14:]!='.dat': raise ValueError('parse error at file name: '+fn)
            sen2=fn[13]
            seni=['1','2']

    elif db=='cfk':
        if len(dn)!=8: raise ValueError('parse error at dir name: '+dn)
        if fn[14]!='_': raise ValueError('parse error at file name: '+fn)
        cls1=fn[0:2]
        sen1=fn[2:4]
        exc=fn[4:8]
        idx=fn[9:14]

        if not cls1 in clsmap_cfk: raise ValueError('class unkown: '+cls1)
        cls1=clsmap_cfk[cls1]
        if not exc in excmap_cfk: raise ValueError('exc unknown: '+exc)
        exc=excmap_cfk[exc]

        if fn[-4:]=='.wav':
            if len(fn)!=21: raise ValueError('parse error at file name: '+fn)
            sen2=fn[15:17]
            seni=['']
        else:
            if len(fn)!=20: raise ValueError('parse error at file name: '+fn)
            if fn[16:]!='.dat': raise ValueError('parse error at file name: '+fn)
            sen2=fn[15]
            seni=['1','2']
    else: raise ValueError("db")


    if not cls1 in fns: fns[cls1]={}
    fnsc=fns[cls1]
    if not exc in fnsc: fnsc[exc]={}
    fnse=fnsc[exc]
    for si in seni:
        sen=sen1+sen2+si
        if not sen in fnse: fnse[sen]={}
        fnss=fnse[sen]
        if not idx in fnss: fnss[idx]=[]
        fnsi=fnss[idx]
        fnsi.append(fno)

ls=0
for cls,fnsc in sorted(fns.items()):
    for exc,fnse in sorted(fnsc.items()):
        lens={}
        for sen,fnss in fnse.items():
            l=len(fnss)
            if not l in lens: lens[l]=[]
            lens[l].append(sen)
            ls+=l
        print("%s: #sen:%i %s"%(cls if exc=='' else cls+'-'+exc,len(fnse),
            str.join('; ',['%i:%s'%(l,str.join(',',sorted(lens[l])) if len(lens[l])<10 else '#%i'%(len(lens[l]))) for l in sorted(lens.keys())])
        ))
print("#%i"%(ls))

if len(sys.argv)>=3 and sys.argv[2]=='-ln':
    def link(src,cls,exc,sen1,sen2,idx):
        if db=='als':
            dst=dbdir+'/common/sig/%s/%03ixx/%s-X1_%s.%s%s.wav'%(
                cls,int(idx)//100,
                cls,idx,sen1,sen2,
            )
        elif db=='cfk':
            dst=dbdir+'/common/sig/%s/%s/%03ixx/%s-%s_%s.%s%s.wav'%(
                cls,exc,int(idx)//100,
                cls,exc,idx,sen1,sen2,
            )
        dd=os.path.dirname(dst)
        if not os.path.exists(dd): os.makedirs(dd)
        if os.path.exists(dst): os.unlink(dst)
        os.link(src,dst)

    for cls,fnsc in sorted(fns.items()):
        for exc,fnse in sorted(fnsc.items()):
            for sen,fnss in fnse.items():
                sen1=sen[:2]
                sen2=sen[2:]
                if sen1==sen2: continue
                for idx,fnsi in fnss.items():
                    for fn in fnsi:
                        if fn[-4:]!='.wav': continue
                        link(fn,cls,exc,sen1,sen2,idx)

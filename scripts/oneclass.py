#!/usr/bin/python3

import sys
import isvm
import ihmm
import iktf
import ijob
import ifdb
from ihelp import *

sys.path.append(os.environ['UASR_HOME']+'-py')


def prtres(prob, flst):
    oklab = list(set(map(lambda x: x['lab'], flst)))
    oklab.sort()
    eer, cm = icls.eer(prob, flst=flst, okpat=oklab[0]) #good label
    return 'EER: %6.2f%% CM: %6.2f%%' % (eer*100, cm*100)


def svmtrn(ftrn, ftst, fea, s, kwargs={}):
    print('svm start  '+s)
    csvm=isvm.trn(ftrn, fea, **kwargs)
    prob=isvm.evlp(csvm, ftst, fea)[:, 0]
    print('svm finish '+s+' '+prtres(prob, ftst))
    return prob


def hmmtrn(ftrn, ftst, fea, s, kwargs={}):
    print('hmm start  '+s)
    chmm = ihmm.trn(flst=ftrn, fea=fea, **kwargs)
    nld = ihmm.evlp(chmm, flst=ftst, fea=fea)
    prob = -nld.take(0, -1)/np.array([f[fea].shape[0] for f in ftst])
    print('hmm finish '+s+' '+prtres(prob,ftst))
    return np.concatenate((prob.reshape(-1, 1), nld), axis=1)


def ktftrn(ftrn, ftst, fea, s, kwargs={}):
    print('dnn start  '+s)
    ktf = iktf.ModKeras(**kwargs)
    ktf.trn(ftrn, fea)
    prob, bc = ktf.evl(ftst, fea, prob=True)
    # probability of class 0
    if 'AEC' not in bc:
        prob = 1-prob[:, 0]
        print('ktf stop  '+s+' '+prtres(prob, ftst))
    else:
        prob = 1 - prob
    return prob

if len(sys.argv) < 2:
    raise ValueError("Usage: "+sys.argv[0]+" CFG [-n]")
icfg.Cfg(*sys.argv[1:])

if '-nn' in sys.argv:
    raise SystemExit()

print("flst")
ftrn = icfg.readflst('train')
ftst = icfg.readflst('test')

dmod = icfg.getdir('model')
if not os.path.exists(dmod):
    os.mkdir(dmod)
dlog=icfg.getdir('log')
if not os.path.exists(dlog):
    os.mkdir(dlog)
dsig = icfg.getdir('sig')
sigext = '.'+icfg.get('sig.ext', 'wav')

sen = getsensors()

senuse = sen
feause = ['pfa','sfa','sig']
clsuse = ['hmm','svm','cnnb','cnn','dnn','ktf']
if not icfg.get('senuse') is None:
    senuse = icfg.get('senuse').split(',')
if not icfg.get('feause') is None:
    feause = icfg.get('feause').split(',')
if not icfg.get('clsuse') is None:
    clsuse = icfg.get('clsuse').split(',')

labmap = {}
labmap = getlabmaps()
if not icfg.get('labmap') is None:
    labmap = icfg.get('labmap')
    labmap = eval(labmap)

maxjobs=16

# the first label is of interest
if labmap is not None:
    ftrn = ftrn.maplab(labmap)
    ftst = ftst.maplab(labmap)

if senuse is not None:
    ftrns = {}
    for strn in senuse:
        ftrns[strn] = []
        ftrns[strn] = ftrn.expandsensor(sen)
        for f in ftrns[strn]:
            f['lab'] = labmap[list(labmap.keys())[0]] if f['sen']==strn else labmap[list(labmap.keys())[1]]
        ftrns[strn] = ftrns[strn].equalcls()
    ftsts = {s: ftst.expandsensor(s) for s in senuse}
else:
    sen = senuse = [icfg.get('db')]
    ftrns = {}
    ftsts = {}
    ftrns[sen[0]] = ftrn
    ftsts[sen[0]] = ftst

fdb = ifdb.Fdb()
for typ in ['sig','pfa']:
    print(typ)
    fdb.analyse(typ, eval(typ+'get'), flst=sum(ftrns.values(), [])+sum(ftsts.values(),[]), jobs=maxjobs)
fdb.save()

print('sfa')
do = set(s for s in senuse if any(not 'sfa' in f for f in ftrns[s]+ftsts[s]))
if len(do) == 1 or maxjobs == 1:
    for s in do:
        sfaget(ftrns[s], ftsts[s], fdb)
else:
    thr = ijob.Thr(maxjobs)
    for s in do:
        if os.path.exists('stop'):
            thr.cleanup()
            raise SystemExit()
        print('sfa start  '+s)
        thr.start('sfa_'+s, sfaget,(ftrns[s], ftsts[s], fdb))
    for s in do:
        thr.res('sfa_'+s)

if '-n' in sys.argv:
    raise SystemExit()

print('cls')
for cls in clsuse:
    for fea in feause:
        if cls == 'hmm' and ftrns[senuse[0]][0][fea].shape[-1] > 40:
            continue
        probfn = os.path.join(dlog, 'prob_'+cls+'_'+fea+'.npy')

        kwargs = icfg.get('trnargs.%s.%s' % (cls, fea))
        if kwargs is None:
            kwargs = {}
            if cls == 'hmm':
                split, its = getsplits()
                classes, states = getstates(labmap)
                if its is not None:
                    kwargs['its'] = its
                if states is not None:
                    kwargs['states'] = states
        else:
            print('trnargs = ' + kwargs)
            kwargs = eval(kwargs)
        fnctrn = eval(cls+'trn')
        if len(senuse) == 1 or maxjobs == 1 or cls=='ktf':
            prob = [fnctrn(ftrns[s], ftsts[s], fea, s, kwargs) for s in senuse]
        else:
            job = ijob.Thr(maxjobs)
            for s in senuse:
                if os.path.exists('stop'):
                    job.cleanup()
                    raise SystemExit()
                job.start(cls+'trn_'+s, fnctrn, (ftrns[s], ftsts[s], fea, s, kwargs))
            prob = [job.res(cls+'trn_'+s) for s in senuse]

        if len(prob) > 0:
            np.save(probfn, prob)

raise SystemExit()


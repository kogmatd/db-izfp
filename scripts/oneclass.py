#!/usr/bin/python3

import sys
import os
sys.path.append(os.environ['UASR_HOME']+'-py')
import isvm
import ihmm
import iktf
import ijob
import ifdb
from ihelp import *


def prtres(prob, flst):
    oklab = list(set(map(lambda x: x['lab'], flst)))
    oklab.sort()
    eer, cm = icls.eer(prob, flst=flst, okpat=oklab[0])
    return 'EER: %6.2f%% CM: %6.2f%%' % (eer*100, cm*100)


def svmtrn(ftrn, ftst, fea, s, args):
    print('svm start  '+s)
    csvm = isvm.trn(ftrn, fea, **args)
    prob = isvm.evlp(csvm, ftst, fea)[:, 0]
    print('svm finish '+s+' '+prtres(prob, ftst))
    return prob


def hmmtrn(ftrn, ftst, fea, s, args):
    print('hmm start  '+s)
    chmm = ihmm.trn(flst=ftrn, fea=fea, **args)
    nld = ihmm.evlp(chmm, flst=ftst, fea=fea)
    prob = -nld.take(0, -1)/np.array([ft[fea].shape[0] for ft in ftst])
    print('hmm finish '+s+' '+prtres(prob, ftst))
    return np.concatenate((prob.reshape(-1, 1), nld), axis=1)


def ktftrn(ftrn, ftst, fea, s, args):
    print('dnn start  '+s)
    ktf = iktf.ModKeras(**args)
    ktf.trn(ftrn, fea)
    prob, bc = ktf.evl(ftst, fea, prob=True)
    # probability of class 0
    if 'AEC' not in bc:
        prob = 1-prob[:, 0]
        print('ktf stop  '+s+' '+prtres(prob, ftst))
    else:
        prob = 1 - prob
        print('ktf stop  ' + s + ' ' + prtres(prob, ftst))
    return prob


snntrn = ktftrn
aectrn = ktftrn
rnntrn = ktftrn
cnntrn = ktftrn

isnn = iktf
iaec = iktf
irnn = iktf
icnn = iktf


if len(sys.argv) < 2:
    raise ValueError("Usage: "+sys.argv[0]+" CFG [-n]")
icfg.Cfg(*sys.argv[1:])

if '-nn' in sys.argv:
    raise SystemExit()

print("flst")
ftrain = icfg.readflst('train')
ftest = icfg.readflst('test')

dmod = icfg.getdir('model')
if not os.path.exists(dmod):
    os.mkdir(dmod)
dlog = icfg.getdir('log')
if not os.path.exists(dlog):
    os.mkdir(dlog)
dsig = icfg.getdir('sig')
sigext = '.'+icfg.get('sig.ext', 'wav')

sen = getsensors()

senuse = sen
feause = ['pfa', 'sfa', 'sig']
clsuse = ['hmm', 'svm', 'ktf']
if not icfg.get('senuse') is None:
    senuse = icfg.get('senuse').split(',')
if not icfg.get('feause') is None:
    feause = icfg.get('feause').split(',')
if not icfg.get('clsuse') is None:
    clsuse = icfg.get('clsuse').split(',')

labmap = getlabmaps()
if not icfg.get('labmap') is None:
    labmap = icfg.get('labmap')
    labmap = eval(labmap)

maxjobs = 16

# the first label is of interest
if labmap is not None:
    ftrain = ftrain.maplab(labmap)
    ftest = ftest.maplab(labmap)

if senuse is not None:
    ftrns = {}
    for strn in senuse:
        ftrns[strn] = []
        ftrns[strn] = ftrain.expandsensor(sen)
        for f in ftrns[strn]:
            f['lab'] = labmap[list(labmap.keys())[0]] if f['sen'] == strn else labmap[list(labmap.keys())[1]]
        ftrns[strn] = ftrns[strn].equalcls()
    ftsts = {s: ftest.expandsensor(s) for s in senuse}
else:
    sen = senuse = [icfg.get('db')]
    ftrns = dict()
    ftsts = dict()
    ftrns[sen[0]] = ftrain
    ftsts[sen[0]] = ftest

fdb = ifdb.Fdb()
for typ in ['sig', 'pfa']:
    print(typ)
    fdb.analyse(typ, eval(typ+'get'), flst=sum(ftrns.values(), [])+sum(ftsts.values(), []), jobs=maxjobs)
fdb.save()

print('sfa')
do = set(s for s in senuse if any('sfa' not in f for f in ftrns[s]+ftsts[s]))
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
        thr.start('sfa_'+s, sfaget, (ftrns[s], ftsts[s], fdb))
    for s in do:
        thr.res('sfa_'+s)

if '-n' in sys.argv:
    raise SystemExit()

print('cls')
for cls in clsuse:
    for feature in feause:
        if cls == 'hmm' and ftrns[senuse[0]][0][feature].shape[-1] > 40:
            continue
        probfn = os.path.join(dlog, 'prob_' + cls + '_' + feature + '.npy')

        kwargs = icfg.get('trnargs.%s.%s' % (cls, feature))
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
            kwargs['type'] = cls
        fnctrn = eval(cls+'trn')
        if len(senuse) == 1 or maxjobs == 1 or cls == 'ktf':
            probability = [fnctrn(ftrns[s], ftsts[s], feature, s, kwargs) for s in senuse]
        else:
            job = ijob.Thr(maxjobs)
            for s in senuse:
                if os.path.exists('stop'):
                    job.cleanup()
                    raise SystemExit()
                job.start(cls+'trn_'+s, fnctrn, (ftrns[s], ftsts[s], feature, s, kwargs))
            probability = [job.res(cls+'trn_'+s) for s in senuse]

        if len(probability) > 0:
            np.save(probfn, probability)

raise SystemExit()

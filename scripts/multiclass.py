#!/usr/bin/python3

import sys
import isvm
import ihmm
import iktf
import ifdb
from ihelp import *
sys.path.append(os.environ['UASR_HOME']+'-py')


def svmtrn(ftrn, ftst, fea, s, kwargs={}):
    print('svm start  '+fea+'_'+s)
    csvm = isvm.trn(ftrn, fea, **kwargs)
    restrn = isvm.evlp(csvm, ftrn, fea)
    restst = isvm.evlp(csvm, ftst, fea)
    restrnc = isvm.evl(csvm, ftrn, fea)
    reststc = isvm.evl(csvm, ftst, fea)
    if regression:
        print('hmm finish ' + fea + '_' + s)
        icls.labf(ftrn)
        icls.labf(ftst)
        trn = list(map(lambda x: x['labf'], ftrn))
        tst = list(map(lambda x: x['labf'], ftst))
        print('Train mse : ', np.square(np.subtract(np.array(trn), restrn)).mean())
        print('Test mse : ', np.square(np.subtract(np.array(tst), restst)).mean())
    else:
        icls.cmp(ftrn, restrnc, 'SVM Training ' + fea + '_' + s)
        icls.cmp(ftst, reststc, 'SVM Testing ' + fea + '_' + s)
    print('svm finish '+fea+'_'+s)
    return csvm, restrn, restst


def hmmtrn(ftrn, ftst, fea, s, kwargs={}):
    print('hmm start  '+fea+'_'+s)
    chmm = ihmm.trn(flst=ftrn, fea=fea, **kwargs)
    nldtrn = ihmm.evlp(chmm, flst=ftrn, fea=fea)
    nldtst = ihmm.evlp(chmm, flst=ftst, fea=fea)
    if regression:
        icls.labf(ftrn)
        icls.labf(ftst)
        trn = list(map(lambda x: x['labf'], ftrn))
        tst = list(map(lambda x: x['labf'], ftst))
        print('Train mse : ', np.square(np.subtract(np.array(trn), nldtrn)).mean())
        print('Test mse : ', np.square(np.subtract(np.array(tst), nldtst)).mean())
    else:
        restrn = np.array(chmm['cls'])[nldtrn.argmin(axis=1)]
        icls.cmp(ftrn, restrn, 'HMM Training '+fea+'_'+s)
        restst = np.array(chmm['cls'])[nldtst.argmin(axis=1)]
        icls.cmp(ftst, restst, 'HMM Testing ' + fea + '_' + s)
    return chmm, nldtrn, nldtst


def ktftrn(ftrn, ftst, fea, s, kwargs={}):
    print('Keras TF start  ' + s)
    ktf = iktf.ModKeras(**kwargs)
    ktf.trn(ftrn, fea)
    restrn, trncls = ktf.evl(ftrn, fea, prob=True)
    restst, tstcls = ktf.evl(ftst, fea, prob=True)
    if regression:
        icls.labf(ftrn)
        icls.labf(ftst)
        trn = list(map(lambda x: x['labf'], ftrn))
        tst = list(map(lambda x: x['labf'], ftst))
        print('Train mse : ', np.square(np.subtract(np.array(trn), restrn[:,0])).mean())
        print('Test mse : ', np.square(np.subtract(np.array(tst), restst[:,0])).mean())
    else:
        icls.cmp(ftrn, trncls, 'Keras TF Training ' + fea + '_' + s)
        icls.report(ftrn, trncls, verbose=True)
        icls.cmp(ftst, tstcls, 'Keras TF Testing ' + fea + '_' + s)
        icls.report(ftst, tstcls, verbose=True)
    print('Keras TF stop  ' + s)
    return ktf.mod, restrn, restst


if len(sys.argv) < 2:
    raise ValueError("Usage: "+sys.argv[0]+" CFG [-n]")
icfg.Cfg(*sys.argv[1:])

ftrn = icfg.readflst('train')
ftst = icfg.readflst('test')

dmod = icfg.getdir('model')
if not os.path.exists(dmod):
    os.mkdir(dmod)
dlog = icfg.getdir('log')
if not os.path.exists(dlog):
    os.mkdir(dlog)

sen = getsensors()
regression = icfg.get('trn.regression') == True

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

maxjob = 16

if '-nn' in sys.argv:
    raise SystemExit()


def run_sen(s):
    print("flst [%s]" % s) if s is not None else print("flst [%s]" % icfg.get('db'))
    fdb = ifdb.Fdb(s)
    ftrns = ftrn
    ftsts = ftst
    if s is not None:
        ftrns = ftrns.expandsensor(s, fdb)
        ftsts = ftsts.expandsensor(s, fdb)
    if labmap is not None:
        ftrns = ftrns.maplab(labmap)
        ftsts = ftsts.maplab(labmap)
    for typ in ['sig', 'pfa']:
        fdb.analyse(typ, eval(typ + 'get'), ftrns + ftsts)
    sfaget(ftrns, ftsts, fdb)
    fdb.save()

    if s is None:
        s = icfg.get('db')
    if not regression:
        ftrns = ftrns.equalcls()
    if '-n' in sys.argv:
        return

    for cls in clsuse:
        for fea in feause:
            if cls == 'hmm' and ftrns[0][fea].shape[-1] > 40:
                continue
            resfn = os.path.join(dlog, 'res_'+cls+'_'+fea+'_'+s+'.npy')

            print('####################', fea, cls, s, '####################')
            kwargs = icfg.get('trnargs.%s.%s' % (cls, fea))
            if kwargs is None:
                kwargs = {}
                if cls == 'hmm':
                    split, its = getsplits()
                    classes, states = getstates()
                    if its is not None:
                        kwargs['its'] = its
                    if states is not None:
                        kwargs['states'] = states
            else:
                print('trnargs = '+kwargs)
                kwargs = eval(kwargs)
            kwargs['regression'] = regression
            fnctrn = eval(cls[:3]+'trn')
            for i in range(3):
                (mod, restrn, restst) = fnctrn(ftrns, ftsts, fea, s, kwargs)

                if len(restst) > 0:
                    np.save(resfn[:-4] + '_tst.npy', restst)
                if len(restrn) > 0:
                    np.save(resfn[:-4] + '_trn.npy', restrn)
                if mod is not None:
                    modfn = os.path.join(dmod, cls + '_' + fea + '_' + s + '.model')
                    eval('i' + cls[:3] + '.save')(mod, modfn)
                break


if senuse is None:
    run_sen(senuse)
elif len(senuse) == 1:
    run_sen(senuse[0])
else:
    # job=ijob.Job(1 if len([cls for cls in clsuse if cls=='dnn' or cls[:3]=='cnn'])>0 else maxjob)
    for s in senuse:
        run_sen(s)
    #    if os.path.exists('stop'): job.cleanup(); raise SystemExit()
    #    job.start('run_'+s,run_sen,(s,))
    # for s in senuse: job.res('run_'+s)

raise SystemExit()
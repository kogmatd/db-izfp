#!/usr/bin/python3

import sys
import os
sys.path.append(os.environ['UASR_HOME']+'-py')
import idat
import ilinreg
from ihelp import *
from ihlp import *

def cor(x, y):
    a = x-np.mean(x)
    b = y-np.mean(y)
    if np.max(np.abs(a)) == 0:
        return 0
    if np.max(np.abs(b)) == 0:
        return 0
    return np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))


def mse(x, y):
    return np.mean((x-y)**2)


def plot_reg(labb, mress, name):
    res_reg ='res/reg/'
    fnn = res_reg + name + '.plot'
    if not os.path.exists(res_reg):
        os.mkdir(res_reg)

    labc = rle(labb)
    pl = [(l, np.mean(mress[s1:s1+e]), np.std(mress[s1:s1+e])) for s1, e, l in labc]
    with open(fnn, 'w') as f:
        f.write('#-title '+name+'\n')
        f.write('#-xlabel Referenz\n')
        f.write('#-ylabel Ergebnis\n')
        f.write('#-col 2\n')
        f.write('#-typ yerrorlines\n')
        for p in pl:
            f.write('%i %.3f %.3f\n' % p)


if len(sys.argv) < 2:
    raise ValueError("Usage: "+sys.argv[0]+" (CFG prob_*.npy | csv)")
# if sys.argv[1]=='csv':
#    for cfgi in range(1,4):
#        cfg='../cfk/X%i/info/default.cfg'%cfgi
#        os.system(str.join(' ',['python3',sys.argv[0],cfg,'csv']))
#    raise SystemExit()
icfg.Cfg(sys.argv[1])
ocsv = len(sys.argv) > 2 and sys.argv[2] == 'csv'
if ocsv:
    sys.argv.remove('csv')

ftst = icfg.readflst('test')
ftrn = icfg.readflst('train')
dlog = icfg.getdir('log')
sen = getsensors()
regression = icfg.get('trn.regression') == True

labmap = getlabmaps()
if not icfg.get('labmap') is None:
    labmap = icfg.get('labmap')
    labmap = eval(labmap)
if labmap is not None:
    ftst.maplab(labmap)
lcls = np.array(ftst.getcls())

if regression:
    icls.labf(ftst)
    lab = np.array([f['labf'] for f in ftst])
    lfcls = [i[-1] for i in rle(sorted(lab))]

resh = dict()
for fn in argv2resfns('res_', sys.argv[2:]):
    if fn.find('_trn.npy') >= 0 or fn.find('.model') >= 0:
        continue
    cls, fea, s, exp = os.path.basename(fn)[4:-4].split('_')
    res = np.load(fn)
    if res.shape == (0,):
        continue
    h = {'res': res}
    if os.path.exists(fn[:-4]+'_trn.npy'):
        h['res_trn'] = np.load(fn[:-4]+'_trn.npy')
    if regression:
        h['c'] = 0
    else:
        if cls == 'hmm':
            res = -res
        if res.shape[1] == 1:
            h['resc'] = resc = lcls[np.round(res[:, 0]).astype(int)]
        else:
            h['resc'] = resc = lcls[res.argmax(axis=-1)]
        h['c'] = np.sum(resc == [f['lab'] for f in ftst])/len(ftst)
    if cls not in resh:
        resh[cls] = dict()
    if fea not in resh[cls]:
        resh[cls][fea] = dict()
    if s in resh[cls][fea]:
        raise ValueError("duplicate")
    resh[cls][fea][s] = h

feas = sorted({fea for rc in resh.values() for fea in rc.keys()})
sens = sorted({s for rc in resh.values() for rf in rc.values() for s in rf.keys()})

# feas=['pfa']

for cls, feah in list(resh.items()):
    for fea, senh in feah.items():
        res_trn = [h['res_trn'].flatten() for h in senh.values() if 'res_trn' in h]
        if len(res_trn) < 10:
            continue
        res_tst = [h['res'].flatten() for h in senh.values() if 'res_trn' in h]
        res_trn = np.array(res_trn).transpose()
        res_tst = np.array(res_tst).transpose()
        ftrnc = [{'lab': f['lab'], 'fea':idat.Dat(rt)} for f, rt in zip(ftrn, res_trn)]
        icls.labf(ftrnc)
        ftstc = [{'fea': idat.Dat(rt)} for rt in res_tst]
        lr = ilinreg.trn(ftrnc)
        lres = ilinreg.evl(lr, ftstc)
        if cls+'_lr' not in resh:
            resh[cls+'_lr'] = {}
        resh[cls+'_lr'][fea] = {'XX': {'res': lres, 'c': 0}}
           

for fea in feas:
    for cls in sorted(resh):
        if fea not in resh[cls]:
            continue
        c = []
        s = []
        res = []
        for si, ri in resh[cls][fea].items():
            s.append(si)
            c.append(ri['c'])
            res.append(ri['res'])
        if regression:
            res = np.array(res)
            if res.shape[-1] == 1:
                res = res.reshape(res.shape[:-1])
            mres = np.mean(res, axis=0)
            acor = cor(mres, lab)
            amse = mse(mres, lab)
            lmse = [mse(mres[lab == l], l) for l in lfcls]
            eer, cm = icls.eer(1-mres, flst=ftst, okpat='Z00')
            print('%s %-7s [%3i/%3i] MEAN cor: %.3f mse: %5.2f max-mse: %5.2f/%s Z00/1-mse: %5.2f, %5.2f EER: %6.2f%% CM: %6.2f%%' % (
                fea, cls, len(resh[cls][fea]), len(sen),
                acor, amse, np.max(lmse), lcls[np.argmax(lmse)], *lmse[:2],
                eer*100, cm*100
            ))
            bcor = [cor(r, lab) for r in res]
            bmse = [mse(r, lab) for r in res]
            plot_reg(lab, mres, fea+'_'+cls)
        else:
            cmax = np.max(c)
            smax = np.array(s)[np.argmax(c)]
            res = np.mean(res, axis=0)
            resc = lcls[res.argmax(axis=-1)]
            cmix = np.sum(resc == [f['lab'] for f in ftst])/len(ftst)
            if cmix < 1:
                cmx = dict()
                for lref in lcls:
                    cmx[lref] = dict()
                    for lres in lcls:
                        cmx[lref][lres] = np.sum(resc[np.array([f['lab'] == lref for f in ftst])] == lres)
                        cmx[lref][lres] /= np.sum([f['lab'] == lref for f in ftst])
                cmxmax = np.max([cmx[lref][lres] for lref in lcls for lres in lcls if lres != lref])
                cmxmaxl = str.join(' ', [lref+'=>'+lres for lref in lcls for lres in lcls if lres != lref
                                         and cmx[lref][lres] == cmxmax])
                msg = ' MAX-ER: %5.1f%% (%s)' % (cmxmax*100, cmxmaxl)
                cmmin = None
            else:
                cma = []
                for l1i in range(len(lcls)):
                    for l2i in range(l1i+1, len(lcls)):
                        if l1i == l2i:
                            continue
                        l1 = lcls[l1i]
                        l2 = lcls[l2i]
                        r = res[:, l1i]-res[:, l2i]
                        r1 = r[np.array([f['lab'] == l1 for f in ftst])]
                        r2 = r[np.array([f['lab'] == l2 for f in ftst])]
                        r1m = r1.mean()
                        r2m = r2.mean()
                        if r1m > r2m:
                            cm = (r1.min()-r2.max())/(r1m-r2m)
                        else:
                            cm = (r2.min()-r1.max())/(r2m-r1m)
                        cma.append((cm*100, l1, l2))
                cmmin = cma[np.array(cma)[:, 0].argmin()]
                msg = ' MAX-ER:   0.0%% MIN-CM %5.1f%% (%s<=>%s)' % cmmin
                cmxmax = 0
                cmxmaxl = '---'
            misstrain = len([1 for ri in resh[cls][fea].values() if ri['res'].max() == ri['res'].min()])
            if misstrain > 0:
                msg += ' MISSTRAIN: %i' % misstrain
            if ocsv:
                print('"%s" "%s" "%s" %s "%s" %s %s "%s" %.1f%% "%s"' % (
                    icfg.get('db').split('/')[-1]+'/'+icfg.get('exp'),
                    fea, cls,
                    '%.1f%%' % ((1-cmix)*100) if len(resh[cls][fea]) == len(sens) else '"RUN"',
                    cmxmaxl if cmxmaxl != '---' else '%s<=>%s' % cmmin[1:],
                    '%.1f%%' % (cmxmax*100) if cmxmaxl != '---' else '"---"',
                    '%.1f%%' % (cmmin[0]) if cmxmaxl == '---' else '"---"',
                    smax, (1-cmax)*100,
                    'MISSTRAIN: %i' % misstrain if misstrain > 0 else '',
                ))
            else:
                print('%s %4s [%3i] best %5.1f%%/%s mix %5.1f%%%s' % (
                    fea, cls, len(resh[cls][fea]),
                    cmax*100, smax,
                    cmix*100, msg
                ))
    print('')

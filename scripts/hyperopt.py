#!/usr/bin/python3

import sys
import os
sys.path.append(os.environ['UASR_HOME']+'-py')

import ifdb
from ihelp import *

# ensure reproducability
import numpy as np
import tensorflow as tf
import random as rn

np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.optimizers import *
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from keras.activations import *
from keras.losses import *
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import talos as ta
from talos.model.layers import hidden_layers
from talos.metrics.keras_metrics import fmeasure_acc
import icls



def dataset(flst, fea, lab, clstype='snn', regression=False):
    '''
    Prepare the dataset and reshape to
    to 3D tensors.
    '''

    scaler = MinMaxScaler(feature_range=(-1, 1))

    trn_array = np.array(list(map(lambda x: x[fea].dat, flst)))

    '''Always in the format (samples, timesteps, features)'''

    if len(trn_array.shape) == 2:
        trn_array = trn_array.reshape(trn_array.shape[0], trn_array.shape[1], 1)

    dims = trn_array.shape
    s0, s1, s2 = dims[0], dims[1], dims[2]
    x_train = trn_array.reshape(s0 * s1, s2)

    scaler.fit(x_train)
    # apply transform
    x_train = scaler.transform(x_train)

    '''Convolutional input needs 1D (batch, steps, channels) or
        2D (batch, rows, cols, channels)
       Dense input layer works best with 1D input 
    '''
    if clstype == 'cnn':
        trn_data = x_train.reshape(s0, s1, s2, 1)
    elif clstype == 'snn':
        trn_data = x_train.reshape(s0, s1 * s2)
    else:
        trn_data = x_train.reshape(s0, s1, s2)

    del trn_array, x_train

    if regression:
        icls.labf(flst)
        trn_labels = list(map(lambda x: x['labf'], flst))
        trn_labels = np.array(trn_labels)
    else:
        trn_labels = list(map(lambda x: x[lab], flst))
        encoder = LabelBinarizer()
        trn_labels = encoder.fit_transform(trn_labels)
        del encoder

    return trn_data, trn_labels


def snnopt(ftrn, ftst, fea, s, kwargs):

    lab = 'lab'
    X, Y = dataset(ftrn, fea, lab, **kwargs)

    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.33, random_state=42)

    def test_model(x_train, y_train, x_val, y_val, params):

        model = Sequential()
        model.add(Dense(params['first_neuron'], input_dim=x_train.shape[1],
                        activation=params['activation'],
                        kernel_initializer=params['kernel_initializer']))

        model.add(Dropout(params['dropout']))

        ## hidden layers
        for i in range(params['hidden_layers']):
            print(f"adding layer {i + 1}")
            model.add(Dense(params['hidden_neuron'], activation=params['last_activation'],
                            kernel_initializer=params['kernel_initializer']))
            model.add(Dropout(params['dropout']))

        #hidden_layers(model, params, 1)

        model.add(Dense(y_train.shape[1], activation=params['last_activation'],
                        kernel_initializer=params['kernel_initializer']))

        model.compile(loss=params['losses'],
                      optimizer=params['optimizer'](),
                      metrics=['acc', fmeasure_acc])

        history = model.fit(x_train, y_train,
                            validation_data=[x_val, y_val],
                            batch_size=params['batch_size'],
                            epochs=params['epochs'],
                            verbose=0)
        return history, model

    # then we can go ahead and set the parameter space
    p = {'first_neuron': [300, 600],
         'hidden_neuron': [300, 600],
         'hidden_layers': [0, 1, 2],
         'batch_size': [265],
         'epochs': [5],
         'dropout': [0.5],
         'kernel_initializer': ['uniform', 'normal'],
         'optimizer': [Adam],
         'losses': [categorical_crossentropy],
         'activation': [relu],
         'last_activation': ['softmax']}

    t = ta.Scan(x=x_train,
                y=y_train,
                model=test_model,
                params=p,
                dataset_name='snn',
                experiment_no=fea,
                print_params=True)

    return None, None


def aecopt(x_train, y_train, params):
    history = None
    model = None
    return history, model


def rnnopt(x_train, y_train, params):
    history = None
    model = None
    return history, model


def cnnopt(x_train, y_train, params):
    history = None
    model = None
    return  history, model


if len(sys.argv) < 2:
    raise ValueError("Usage: "+sys.argv[0]+" CFG [-n]")
icfg.Cfg(*sys.argv[1:])

ftrain = icfg.readflst('train')
ftest = icfg.readflst('test')

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
clsuse = ['snn', 'cnn', 'rnn', 'aec']
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
    ftrns = ftrain
    ftsts = ftest
    if s is not None:
        ftrns = ftrns.expandsensor(s, fdb)
        ftsts = ftsts.expandsensor(s, fdb)
    if labmap is not None:
        ftrns = ftrns.maplab(labmap)
        ftsts = ftsts.maplab(labmap)

    trn_labels = list(map(lambda x: x['lab'], ftrns))
    trn_nclasses = len(set(trn_labels))

    if trn_nclasses < 2 and not regression:
        raise ValueError("Multiclass classification cannot be trained on one class")

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
            resfn = os.path.join(dlog, 'res_'+cls+'_'+fea+'_'+s+'.npy')
            print('####################', fea, cls, s, '####################')
            kwargs = icfg.get('optargs.%s.%s' % (cls, fea))
            if kwargs is None:
                kwargs = dict()
            else:
                print('optargs = '+kwargs)
                kwargs = eval(kwargs)
            kwargs['regression'] = regression
            kwargs['clstype'] = cls
            fnctrn = eval(cls[:3]+'opt')
            for i in range(3):
                (hist, mod) = fnctrn(ftrns, ftsts, fea, s, kwargs)

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
    for sns in senuse:
        run_sen(sns)
    #    if os.path.exists('stop'): job.cleanup(); raise SystemExit()
    #    job.start('run_'+s,run_sen,(s,))
    # for s in senuse: job.res('run_'+s)

raise SystemExit()

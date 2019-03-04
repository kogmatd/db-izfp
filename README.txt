##### Database for airplane materials #####

Description of signal acquisition, see 2018_ndt_aero/We_6_A_5_Duckhorn.pptx
    Aluminum plate: Directory als/
    CFRP plate
        Ricker wavelet 100kHz: Directory cfk/X1*
        Ricker wavelet 350kHz: Directory cfk/X2*
        Sinc 600kHz: Directory cfk/X3*

The signals you can find under:
	S:\Ablage_IKTS-340\_Projekte\Projekte_ohne_PN\345-DBs\uasr-data\izfp

All following training commands support selection of what is done with the options:
    -Psenuse=SENUSE where SENUSE is a comma separated list of sensors (A1A2,A1B1,A1B2,...,D1D2 for als and ...,F1F2 for cfk)
    -Pfeause=FEAUSE where FEAUSE is a comma separated list of features:
        sig: Raw signals
        pfa: Primary feature analysis
        sfa: Secondary feature analysis
    -Pclsuse=CLSUSE where CLSUSE is a comma separated list of classifiers: hmm,svm,dnn,cnn,cnnb

##### Multiclass training and evaluation #####

The following commands will generate feature database files, train the classifier,
evaluate it and save the results (likelihood for every file and every class) in the log directory.

./scripts/multiclass.py ./cfk/X1/info/default.txt [-Psenuse=SENUSE] [-Pfeause=FEAUSE] [-Pclsuse=CLSUSE]
./scripts/multiclass.py ./cfk/X2/info/default.txt [-Psenuse=SENUSE] [-Pfeause=FEAUSE] [-Pclsuse=CLSUSE]
./scripts/multiclass.py ./cfk/X3/info/default.txt [-Psenuse=SENUSE] [-Pfeause=FEAUSE] [-Pclsuse=CLSUSE]

The following commands print the results of the evaluation:

./scripts/multiclass_res.py ./cfk/X1/info/default.txt {RES-FILES}
./scripts/multiclass_res.py ./cfk/X2/info/default.txt {RES-FILES}
./scripts/multiclass_res.py ./cfk/X3/info/default.txt {RES-FILES}

Only the files gives as RES-FILES are processed.
If RES-FILES is omitted, all files named 'res_*' in the log directory are processed.
The following result values are printed:
  feature-type
  classifier-type
  [number of sensors with results]
  best NN%/SEN: Correctness of the sensor SEN with the highest correctness
  mix NN%:      Correctness by using the mean likelihood over all sensors for classification
  MAX-ER NN%:   Worst error rate between two sensors
  MIN-CM NN%:   Minimal safety margin between two sensors
  (SEN<=>SEN)   The two sensors with the worst result

##### Regression ####

The following command will generate feature database files, train the regression,
evaluate it and save the results (estimated value for each file) in the log directory.

./scripts/multiclass.py ./als/common/info/default.txt [-Psenuse=SENUSE] [-Pfeause=FEAUSE] [-Pclsuse=CLSUSE]

The following command print the results of the evaluation:

./scripts/multiclass_res.py ./als/common/info/default.txt {RES-FILES}

Only the files gives as RES-FILES are processed.
If RES-FILES is omitted, all files named 'res_*' in the log directory are processed.
The following result values are printed:
  feature-type
  classifier-type
  [number of sensors with results]
  mse NN:          Mean square error for mean result of all sensors
  max-mse: NN/SEN: Mean square error of sensor SEN with the highest mean square error
  Z00/1-mse NN,NN: Mean square error of Z00 and Z01
  EER:             Equal error rate for mean result of all sensors
  CM:              classification safety margin

##### Oneclass detector ####

The following commands will train a oneclass detector for good state Z00.
The signals of not Z00 states are only used for evaluation.
For classifiers which need at least two classes for training it uses the Z00 data
of all other sensors but the current one as universal background model.
The evaluation results will be saved as likelihood matrix with a likelihood for class Z00 and each file
and each sensor in the log directory.

./scripts/oneclass_ubm.py ./als/oneclass/info/default.txt [-Pfeause=FEAUSE] [-Pclsuse=CLSUSE]
./scripts/oneclass_ubm.py ./cfk/X1-oneclass/info/default.txt [-Pfeause=FEAUSE] [-Pclsuse=CLSUSE]
./scripts/oneclass_ubm.py ./cfk/X2-oneclass/info/default.txt [-Pfeause=FEAUSE] [-Pclsuse=CLSUSE]
./scripts/oneclass_ubm.py ./cfk/X3-oneclass/info/default.txt [-Pfeause=FEAUSE] [-Pclsuse=CLSUSE]

The following command print the results of the evaluation:

./scripts/oneclass_prob_res.py ./als/oneclass/info/default.txt {RES-FILES}
./scripts/oneclass_prob_res.py ./cfk/X1-oneclass/info/default.txt {RES-FILES}
./scripts/oneclass_prob_res.py ./cfk/X2-oneclass/info/default.txt {RES-FILES}
./scripts/oneclass_prob_res.py ./cfk/X3-oneclass/info/default.txt {RES-FILES}

Only the files gives as RES-FILES are processed.
If RES-FILES is omitted, all files named 'res_*' in the log directory are processed.
The following result values are printed:
  file name
  ERR NN: Equal error rate
  CM NN:  Classification safety margin

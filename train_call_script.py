import os 
import sys 
import pickle
import math
import string 
import random
import itertools


BATCH_SZ = [32]#[32,64]#[32,64]
LR = 1e-04
GCSIZE = [64]
LSTMSIZE = [128,64]
DROPOUT = [0.0,0.1]#[0.0,0.2]
DECODER = "LSTMB"
EXPERIMENTTYPE = "BEST RESPLIT"
SPLITTYPE = "NEW"
FEATURISERFLAG = "INTERMEDIATE"
SAVERESULTS = True
WEIGHTDECAY = [0.0001,0.0]
SRC = "DIFFERENT SPLIT"
OPTIMSCHEDULE = "NONE"
SEED = 2
CLIPGRAD = True
CLIPVAL = 5.0
TRACKING = True #TRUE WHEN RUNNING EXPERIMENTS WHICH SHOULD BE LOGGED
DEBUG = False
EARLYSTOPPING = False
NUM_SAMPLES = "ALL"
N_EPOCHS = 150
LSTM_NORMALIZATION = [True]
LSTM_LAYERS = [4,2]
VAL_EVERY = 10
WEIGHTEDSAMPLING = False
WEIGHTMODULATOR = [[2.0,1.5,1.0,1.0,1.0,1.0],
                   [2.0,1.0,1.0,1.0,1.0,1.0],
                   [1.5,1.0,1.0,1.0,1.0,1.0],
                   [1.0,1.0,1.0,1.0,1.0,1.0]]

for x in itertools.product(GCSIZE, LSTMSIZE,LSTM_NORMALIZATION,LSTM_LAYERS,BATCH_SZ,DROPOUT,WEIGHTDECAY): #run through all combinations to check if layernorm and grad clip helps out
    gcsize = x[0]
    lstmsize = x[1]
    lstmnorm = x[2]
    lstmlayers = x[3]
    batch_sz = x[-3]
    dropout = x[-2]
    weightdecay = x[-1]
    #weightmodulator = x[-1]
    weightmodulator = 0

    cfg_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    param_cfg = {"experimentType": EXPERIMENTTYPE,"SplitType":SPLITTYPE,"NUM_SAMPLES":NUM_SAMPLES,"Featuriser":FEATURISERFLAG,
                 "DecoderType":DECODER,"LR":LR*math.sqrt(batch_sz),
                 "WEIGHTDECAY":weightdecay,"GCSize":gcsize,"LSTMSize":lstmsize,"Dropout":dropout,"BATCH_SZ":batch_sz,
                 "N_EPOCHS":N_EPOCHS,"Save results":SAVERESULTS,
                 "SRC":SRC,"OPTIMSCHEDULE":OPTIMSCHEDULE,"SEED":SEED,"TRACKING":TRACKING,"DEBUG":DEBUG,
                 "CLIPGRADS":CLIPGRAD,"EARLYSTOP":EARLYSTOPPING,"CLIPVAL":CLIPVAL,"LSTMNORM":lstmnorm,
                 "LSTMLAYERS":lstmlayers,"VALINTERVAL":VAL_EVERY,"WEIGHTEDSAMPLING":WEIGHTEDSAMPLING,"WEIGHTMODULATOR":weightmodulator}
    path = "experimental_cfg"+"_"+cfg_id+".pkl"
    with open(path, "wb") as f:
        pickle.dump(param_cfg,f) #is overwritten at every iteration
    

    command = "python3 train.py " + str(path)
    os.system(command)
    




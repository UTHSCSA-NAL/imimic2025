#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
import subprocess as sp
import time



def loadFile(fil):
    fi=open(fil,'r')
    r=fi.readlines()
    fi.close()
    for i in range(0,len(r)):
        while r[i].endswith('\n') or r[i].endswith('\r') and len(r[i])>1:
            r[i]=r[i][:-1]
        if r[i]=='\r' or r[i]=='\n':
            r[i]=''
    return r
def saveFile(r,fil):
    fi=open(fil,'w')
    for i in range(0,len(r)):
        fi.write(r[i]+'\n')
    fi.close() 

class DeepMapping():
    def __init__(self,layers,dim_inp,dim,dim_oup,alpha):
        init=keras.initializers.RandomNormal(mean=0.0, stddev=0.05)


        input = keras.layers.Input(shape=(dim_inp,))

        if layers==1:
            last=keras.layers.Dense(dim,activation="softmax",bias_initializer=init,kernel_initializer=init)(input)
        elif layers==2:
            layer_1=keras.layers.Dense(dim,activation="leaky_relu",bias_initializer=init,kernel_initializer=init)(input)
            last=keras.layers.Dense(dim,activation="softmax",bias_initializer=init,kernel_initializer=init)(tf.keras.layers.BatchNormalization()(layer_1))
        elif layers==3:
            layer_1=keras.layers.Dense(dim,activation="leaky_relu",bias_initializer=init,kernel_initializer=init)(input)
            layer_2=keras.layers.Dense(dim,activation="leaky_relu",bias_initializer=init,kernel_initializer=init)(tf.keras.layers.BatchNormalization()(layer_1))
            last=keras.layers.Dense(dim,activation="softmax",bias_initializer=init,kernel_initializer=init)(tf.keras.layers.BatchNormalization()(layer_2))
        elif layers==4:
            layer_1=keras.layers.Dense(dim,activation="leaky_relu",bias_initializer=init,kernel_initializer=init)(input)
            layer_2=keras.layers.Dense(dim,activation="leaky_relu",bias_initializer=init,kernel_initializer=init)(tf.keras.layers.BatchNormalization()(layer_1))
            layer_3=keras.layers.Dense(dim,activation="leaky_relu",bias_initializer=init,kernel_initializer=init)(tf.keras.layers.BatchNormalization()(layer_2))
            last=keras.layers.Dense(dim,activation="softmax",bias_initializer=init,kernel_initializer=init)(tf.keras.layers.BatchNormalization()(layer_3))
        elif layers==5:
            layer_1=keras.layers.Dense(dim,activation="leaky_relu",bias_initializer=init,kernel_initializer=init)(input)
            layer_2=keras.layers.Dense(dim,activation="leaky_relu",bias_initializer=init,kernel_initializer=init)(tf.keras.layers.BatchNormalization()(layer_1))
            layer_3=keras.layers.Dense(dim,activation="leaky_relu",bias_initializer=init,kernel_initializer=init)(tf.keras.layers.BatchNormalization()(layer_2))
            layer_4=keras.layers.Dense(dim,activation="leaky_relu",bias_initializer=init,kernel_initializer=init)(tf.keras.layers.BatchNormalization()(layer_3))
            last=keras.layers.Dense(dim,activation="softmax",bias_initializer=init,kernel_initializer=init)(tf.keras.layers.BatchNormalization()(layer_4))
        



        head=keras.layers.Dense(dim_oup,activation="linear",kernel_regularizer=tf.keras.regularizers.L2(alpha),bias_initializer=init,kernel_initializer=init)(last)
        self.model=keras.Model(inputs=input, outputs=head)
        self.maps=keras.Model(inputs=input, outputs=last)

    def train(self,xtr,ytr):
        print('      learning rate 0.01')
        self.model.compile(loss="mean_absolute_error",optimizer=keras.optimizers.Adam(learning_rate=0.01))
        self.model.fit(xtr,ytr, batch_size=512, epochs=250, verbose=0)
        print('      learning rate 0.001')
        self.model.compile(loss="mean_absolute_error",optimizer=keras.optimizers.Adam(learning_rate=0.001))
        self.model.fit(xtr,ytr, batch_size=512, epochs=250, verbose=0)
        print('      learning rate 0.0001')
        self.model.compile(loss="mean_absolute_error",optimizer=keras.optimizers.Adam(learning_rate=0.0001))
        self.model.fit(xtr,ytr, batch_size=512, epochs=250, verbose=0)
        print('      learning rate 0.00001')
        self.model.compile(loss="mean_absolute_error",optimizer=keras.optimizers.Adam(learning_rate=0.00001))
        self.model.fit(xtr,ytr, batch_size=512, epochs=250, verbose=0)


    def predict(self,data):
        return self.model.predict(data)

    def maps(self,data):
        return self.maps(data)


def modelling(xtr,ytr,layers,dim,alpha):
    model=DeepMapping(layers,xtr.shape[1],dim,ytr.shape[1],alpha)
    model.train(xtr,ytr)
    return model

def core(layers,dim,alpha):
    print('Training models using all data')
    sides={'L':'left','R':'right'}
    for side in ['left','right']:
        print('   '+side)
        xtr=np.load('data_'+side+'_geodesic_distances.npy')
        ytr=np.load('data_'+side+'_quantiles.npy')

        ta=time.time()
        mod=modelling(xtr,ytr,int(layers),int(dim),float(alpha))
        tb=time.time()
        delta=tb-ta
        print('      training time: '+str(delta)+' seconds')

        maps=mod.maps(xtr)
        fmap='maps_'+side+'.npy'
        np.save(fmap,maps)



            
################################################################################
if __name__ == "__main__":
    from argparse import ArgumentParser, RawTextHelpFormatter
    parser = ArgumentParser(description="",formatter_class=RawTextHelpFormatter)

    parser.add_argument("-l", "--layers",help="", required=True)
    parser.add_argument("-d", "--dim",help="", required=True)
    parser.add_argument("-a", "--alpha",help="", required=True)
    
    args = parser.parse_args()
    core(args.layers,args.dim,args.alpha )



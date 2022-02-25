# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 18:36:41 2022

@author: linxiaohua
"""

import numpy as np
from numpy import mat
import pickle
import tensorflow as tf

test=np.load('./test_np.npy',allow_pickle=True)

with open('./peptide_feature_dict','rb') as f:
	peptide_feature_dict = pickle.load(f)

#print(test)
pep_seq, label_list,feature= [], [],[]

datafile='pep_inf.csv'
with open(datafile) as f:
    for line in f.readlines()[1:]:
            seq, label = line.strip().split(',')
            pep_seq.append(seq)
            label_list.append(label)
            #print(peptide_feature_dict[seq],type(peptide_feature_dict[seq]))
            feature.append(peptide_feature_dict[seq])
print(type(feature))    

X_label = np.array(label_list,dtype='uint8')
X_feature = np.array(feature,dtype='uint8')
X_feature = np.reshape(X_feature, (367,50,1))

#X_label = tf.constant(1.) if tf.math.equal(label_list.all(), tf.constant('cpp', dtype=tf.string)) else tf.constant(0.)

from keras.layers import *
from keras.models import *
from tensorflow.keras import optimizers
import tensorflow as tf


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_feature, X_label, test_size = 0.25, random_state = 0)


model = Sequential()

model.add(Input(shape=(50,)))
#model.add(Flatten)                                                                
model.add(Dense(50, activation='relu')) 
model.add(Dropout(0.2))  
model.add(Dense(25, activation='relu'))                                               
model.add(Dense(1, activation='softmax'))              

model.summary()

model.compile(optimizer = 'adam',                    
              loss = tf.losses.binary_crossentropy,    
              metrics = ['accuracy'])

#from tensorflow.keras.utils import to_categorical
#y = to_categorical(y_train)

history = model.fit(x_train, y_train, batch_size=25, 
                    epochs=30, validation_split=0.2)

############################
embedder = Embedding(23, 100, input_length=50)

model = Sequential()
model.add(embedder)  
model.add(Conv1D(256, 3, padding='same', activation='relu')) 
model.add(MaxPool1D(45, 3, padding='same'))           
model.add(Conv1D(32, 3, padding='same',  activation='relu')) 
model.add(Flatten())                                       
model.add(Dropout(0.3))                                     
model.add(Dense(256, activation='relu'))                  
model.add(Dropout(0.2))                                    
model.add(Dense(1, activation='softmax'))           
model.summary()

model.compile(optimizer = 'adam',                    
              loss = tf.losses.binary_crossentropy,     
              metrics = ['accuracy'])

history = model.fit(x_train, y_train, batch_size=256, 
                    epochs=30, validation_split=0.2)
plot(history)
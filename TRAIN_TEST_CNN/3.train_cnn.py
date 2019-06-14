#coding:utf-8   


import random  
import numpy as np
import math
import h5py
import time
import os
from six.moves import range  
import pickle
import sys
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import *  
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation, Flatten  
from keras.layers.advanced_activations import PReLU  
from keras.layers.convolutional import Conv2D, MaxPooling2D,AveragePooling2D
from keras.optimizers import SGD, Adadelta, Adagrad  
from keras.utils import np_utils, generic_utils  
from keras import backend


list_lr=0.001
list_batch=50
list_epoch=500
all_txt=""
train_txt=""
test_txt=""
img_path=''

t0=time.time()
X_train = pickle.load( open(sys.argv[1], 'rb'))
train_label = pickle.load( open(sys.argv[2], 'rb'))
# X_train,train_label= load_data(train_txt,img_path,1,'train')
t1=time.time()
print ("loading data uses time: "+str(t1-t0))

train_label = np_utils.to_categorical(train_label,3)  
print('Build model...')
model = Sequential()  

model.add(Conv2D(8, (3, 3), padding="valid", kernel_initializer="uniform", weights=None, input_shape=(100,100,3)) )
model.add(Activation('relu'))

model.add(Conv2D(8,(3,3),padding="valid")) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
 
model.add(Conv2D(16,(3,3), padding="valid"))
model.add(Activation('relu'))                   
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(16,(2,2), padding="valid"))
model.add(Activation('relu'))   
model.add(Dropout(0.2))

model.add(Flatten()) 
model.add(Dense(64, kernel_initializer="normal"))
model.add(Activation('relu'))

model.add(Dense(3, kernel_initializer="normal")) 
model.add(Activation('softmax'))  
model.summary()  

sgd = SGD(lr=list_lr, decay=0.00001, momentum=0.1, nesterov=False)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=sgd)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
t2=time.time()
model.fit(X_train,train_label, batch_size=list_batch, epochs=list_epoch,shuffle=False,verbose=2,validation_split=0.2,callbacks=[early_stopping])
# model.fit(X_train,train_label, batch_size=list_batch, epochs=list_epoch,shuffle=False,verbose=2,validation_split=0.15)
t3=time.time()
model.save('')

model.summary()  
print ("loading data uses time: "+str(t1-t0))
print ("training data uses time: "+str(t3-t2))
backend.clear_session()

# list_epoch=500
# model.fit(X_train,train_label, batch_size=list_batch, epochs=list_epoch,shuffle=False,verbose=2,validation_split=0.2)
# model.save('model_saved/model_1117_1000.h5')

# list_epoch=500
# model.fit(X_train,train_label, batch_size=list_batch, epochs=list_epoch,shuffle=False,verbose=2,validation_split=0.2)
# model.save('model_saved/model_1117_1500.h5')

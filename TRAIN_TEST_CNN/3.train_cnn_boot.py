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
from keras.callbacks import Callback
from keras.preprocessing.sequence import *  
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation, Flatten  
from keras.layers.advanced_activations import PReLU  
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad  
from keras.utils import np_utils, generic_utils  
import matplotlib.pyplot as plt 
from keras import backend
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type,fig_name):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig(fig_name+'.jpg')
        plt.close()

def make_noise(train_label,precent,seed_ini):
    noise_label= train_label.copy()
    total = np.shape(train_label)[0]
    noise_lines = random.sample(range(seed_ini,total),int(total*float(precent)))
    for i in noise_lines:
        label_list = [0,1,2]
        label_list.remove(train_label[i])
        noise_label[i] = random.choice(label_list)
    return noise_label


def add_layers(model):

    model.add(Conv2D(8, (3, 5), padding="valid", kernel_initializer="uniform", weights=None, input_shape=(100,200,3)) )
    model.add(Activation('relu'))

    model.add(Conv2D(8,(3,5),padding="valid")) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Dropout(0.2))
     
    model.add(Conv2D(16,(3,3), padding="valid"))
    model.add(Activation('relu'))                   
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten()) 
    model.add(Dense(64, kernel_initializer="normal"))
    model.add(Activation('relu'))

    model.add(Dense(3, kernel_initializer="normal")) 
    model.add(Activation('softmax'))  
    model.summary()  

    list_lr=float(sys.argv[5])
    sgd = SGD(lr=list_lr, decay=0.00001, momentum=0.1, nesterov=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=sgd)
    return model

def train(model,X_train,train_label,list_batch,list_epoch, threshod):
    model = add_layers(model)
    history = LossHistory()
    seed = X_train[:200]
    seed_l = train_label[:200]
    rest = X_train
    rest_l = train_label
    exists = [0 for i in X_train]
    for i in range(200):
        exists[i]=1

    model.fit(seed,seed_l, batch_size=list_batch, epochs=list_epoch,shuffle=False,verbose=2,validation_split=0.15,callbacks=[history])
    stop=0
    while stop==0:
        rest_res = model.predict(rest)
        candidates = [max(i) for i in rest_res]
        print('||||||||||||||||||||||||||||||||||||||||||||')
        print(max(candidates))
        print(min(candidates))
        candidates_l = list(rest_res.argmax(axis=1))
        stop_tmp=0
        for index in range(len(candidates)):
            if candidates_l[index] == list(rest_l[index]).index(1)  and  candidates[index] > threshod  and exists[index]==0 : 
                tmp = np.empty([1,100,200,3])
                tmp[0] = rest[index]
                seed =  np.concatenate((seed, tmp))
                tmp = np.empty([1,3])
                tmp[0] = rest_l[index] 
                seed_l = np.concatenate((seed_l,tmp))
                exists[index]=1 
                stop_tmp=1
        if stop_tmp==0:
            stop=1
        else:
            model.fit(seed,seed_l, batch_size=list_batch, epochs=list_epoch,shuffle=False,verbose=2,validation_split=0.15)
    return model,history   

def test(result,test_label):

    j_0_0 = 0
    j_0_1 = 0
    j_0_2 = 0
    j_1_0 = 0
    j_1_1 = 0
    j_1_2 = 0
    j_2_0 = 0
    j_2_1 = 0
    j_2_2 = 0

    for i,pred in enumerate(list(result)):
        res = list(pred)
        label = test_label[i]
        label_pred = res.index(max(res))

        if label==0 and label_pred == 0:
            j_0_0+=1
        elif label==0 and label_pred == 1:
            j_0_1+=1
        elif label==0 and label_pred == 2:
            j_0_2+=1
        elif label==1 and label_pred == 0:
            j_1_0+=1
        elif label==1 and label_pred == 1:
            j_1_1+=1
        elif label==1 and label_pred == 2:
            j_1_2+=1
        elif label==2 and label_pred == 0:
            j_2_0+=1
        elif label==2 and label_pred == 1:
            j_2_1+=1
        elif label==2 and label_pred == 2:
            j_2_2+=1

    print("j_0_0 = " + str(j_0_0))
    print("j_0_1 = " + str(j_0_1))
    print("j_0_2 = " + str(j_0_2))
    print("j_1_0 = " + str(j_1_0))
    print("j_1_1 = " + str(j_1_1))
    print("j_1_2 = " + str(j_1_2))
    print("j_2_0 = " + str(j_2_0))
    print("j_2_1 = " + str(j_2_1))
    print("j_2_2 = " + str(j_2_2))



def main():

    # loading data-----------------------------------------------------------------------------------

    t0=time.time()
    X_train= pickle.load( open(sys.argv[1], 'rb'))
    #X_train_right = pickle.load( open(sys.argv[2], 'rb'))
    train_label = pickle.load( open(sys.argv[2], 'rb'))
    noise = sys.argv[3]
    noise_label = make_noise(train_label,noise,200)
    list_lr=float(sys.argv[4])
    list_batch=int(sys.argv[5])
    list_epoch=int(sys.argv[6])
    X_test = pickle.load( open(sys.argv[7], 'rb'))
    #X_test_right = pickle.load( open(sys.argv[9], 'rb'))
    test_label = pickle.load( open(sys.argv[8], 'rb'))

    model_name='cnn_cover'+'_'+sys.argv[3]+'_'+str(list_lr)+'_'+str(list_batch)+'_'+str(list_epoch)
    #right_model_name='cnn_right'+'_'+sys.argv[4]+'_'+str(list_lr)+'_'+str(list_batch)+'_'+str(list_epoch)
    t1=time.time()
    print ("loading data uses time: "+str(t1-t0))

    noise_label = np_utils.to_categorical(noise_label,3)  
    print('Build model...')
    model = Sequential() 
    threshod = 0.8
    model,history = train(model,X_train,noise_label,list_batch,list_epoch, threshod)
    model.save(''+ model_name+'.h5')
    
    result = model.predict(X_test)
    test(result,test_label)
    
if __name__ == '__main__':
    main()


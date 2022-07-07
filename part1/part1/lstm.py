# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 15:44:27 2021

@author: 86156
"""
import pandas as pd
import numpy as np
from numpy import array
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from keras.models import Sequential
from keras.models import load_model
from sklearn.utils import shuffle
from keras.layers import LSTM
from keras import regularizers
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Dropout,Activation
import time
import matplotlib.pyplot as plt
from urllib import parse
from keras.preprocessing import sequence
import os
from sklearn.preprocessing import LabelBinarizer
from keras import layers
from sklearn.preprocessing import StandardScaler  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ---- 数据导入 ----
data = pd.read_csv('1.csv',header=None)
data.head()

#categorical_columns = ['Flow Duration','Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min']
#for f in categorical_columns:
    #data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
# shuffle(data)
data[data.isnull()].fillna(0,inplace=True)
target =data.iloc[:,-1]
#data=data[categorical_columns]
data = data.iloc[:,5:20]

scaler = StandardScaler()

data=scaler.fit_transform(data) #fit，本质是生成均值和方差 
#通过接口导出结果 



lb = LabelBinarizer()
target = lb.fit_transform(target) # transfer label to binary value

print(target.shape)

#data = data.apply(func, axis=1, result_type='broadcast')

#data = np.reshape(data,(data.shape[0], data.shape[1],1))


# ---- 参数定义----
#x_train,x_test=train_test_split(data,test_size=0.2)
#y_train,y_test=train_test_split(target,test_size=0.2)



print("Data processing is finished!")

# design network
def build_model(max_features, maxlen):
    # 定义Sequential模型结构，是layer的线性堆栈，无分支，直接堆叠
    model = Sequential()#定义
    model.add(Embedding(max_features,128, input_length=15))
    #model.add(LSTM(500,input_shape=(x_train.shape[1], x_train.shape[2])))#添加LSTM神经网络，即在头文件中已经导入了LSTM，输入神经网络数据形状为train_X.shape[1], train_X.shape[2])，其中500为神经元个数
    model.add(LSTM(128))
    model.add(Dropout(0.5))#可根据自身情况添加dropout

   # model.add(Dense(1))#定义输出层神经元个数为1个，即输出只有1维
    model.add( (Dense(1)))
    
    # model.add(Activation('sigmoid'))#根据情况添加激活函数

    
    # 激活函数sigmoid，把输入的连续实值“压缩”到0和1之间
    # 如果是非常大的负数，输出就是0；如果是非常大的正数，输出就是1
    model.add(Activation('sigmoid'))
    model.summary()
    
    # 使用优化器对交叉熵损失函数进行优化，adam是梯度下降算法的一种变形
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def show_train_history(train_history, train, velidation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[velidation])
    plt.title("train history")  # 标题
    plt.xlabel('Epoch')  # x轴标题
    plt.ylabel(train)  # y轴标题
    plt.legend(['train', 'test'], loc='upper left')  # 图例 左上角
    plt.show()
    
def run(max_features, maxlen,x_train,x_test,y_train,y_test):
    model = build_model(max_features, maxlen)  # build_model
    
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=3, batch_size=128, shuffle=True, validation_split=0.1)      # 训练
    end_time = time.time()
    print("花费时间为{}".format(end_time - start_time))
    model.save('u.h5')
    score, acc = model.evaluate(x_test, y_test, batch_size=128)
    # print('评分:', score)
    print('测试集的Accuracy:', acc)

    # 可视化训练过程，使用2.2show_train_history()
    # acc是模型训练精度，val_acc是指模型在测试集上的精度
    show_train_history(history, 'acc', 'val_acc')  # 训练集/测试集的Accuracy 折线图
    show_train_history(history, 'loss', 'val_loss')  # 训练集/测试集的loss 折线图

#valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(data)))}     
max_features =100#特征维度

maxlen = x_train.shape[1]   # 获得输入特征的最大长度
print(maxlen)
#data=process_data(valid_chars,data,maxlen)

# 分隔训练集和测试集
x_train,x_test,y_train,y_test=train_test_split(data,target,random_state=1,test_size=0.2,shuffle=True)
#x_train = tf.expand_dims(x_train,1) 
#y_train = tf.expand_dims(y_train,1)

#y_test=y_test.iloc[:]
#y_train=y_train.iloc[:]
# 开始模型训练
run(max_features, maxlen,x_train,x_test,y_train,y_test)  # run
# 搭建LSTM模型

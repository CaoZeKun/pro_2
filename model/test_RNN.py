import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rnn

data_csv = pd.read_csv('./data.csv', usecols=[1])
plt.plot(data_csv)

# 数据预处理
data_csv = data_csv.dropna()
dataset = data_csv.values
dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: x / scalar, dataset))

def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# 创建好输入输出
data_X, data_Y = create_dataset(dataset)

# 划分训练集和测试集，70% 作为训练集
train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

import torch

train_X = train_X.reshape(-1, 1, 2)
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, 2)

rnn.Flow(data=data,K_fea=4,HIDDEN_SIZE=20,OUTPUT_SIZE=2,PATH=path,num_epochs=10,LR=0.1,isClassfier=True,MODEL='LSTM')













if __name__ =='__main__':
    # load data | construct model | train | save
    data = np.loadtxt('../data/iris.data',delimiter=',')  # two class

    # print(np.shape(data))
    # path0 = '/model_save/model_params.pkl'
    # path1 = '/model_save/model_params.pkl'
    path = './model_save/model_params.pkl'
    data_test = data[:,:4]


    Flow(data=data,K_fea=4,HIDDEN_SIZE=20,OUTPUT_SIZE=2,PATH=path,num_epochs=10,LR=0.1,isClassfier=True,MODEL='LSTM')

    # load model | predict/test
    # data_test should only have feature
    data_x, data_y = load_model_test(path,data_test,isClassfier=True)
    print(data_y)
    print(data[:,4])
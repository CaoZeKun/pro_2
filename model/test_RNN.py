import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import RNN1D as rnn

data_csv = pd.read_csv('../data/data.csv', usecols=[1])
plt.plot(data_csv)
plt.show()
# 数据预处理
data_csv = data_csv.dropna()
dataset = data_csv.values
dataset = dataset.astype('float16')
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

    # data = np.concatenate((dataX,dataY),axis=0)
    return np.array(dataX), np.array(dataY)

data_X, data_Y = create_dataset(dataset)
train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]


def create_dataset1(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        a.extend(dataset[i + look_back])
        dataX.append(a)
        # dataY.append(dataset[i + look_back])

    dataX = np.array(dataX).astype('float16')
    # dataY = np.array(dataY)
    # print(np.shape(dataX),np.shape(dataY))
    # data = np.concatenate((dataX,dataY),axis=0)
    return dataX



path = './model_save/model_params_air.pkl'

data = create_dataset1(dataset)
print(np.shape(data))

data_y, pred_y = rnn.Flow(data=data,Seq=1,window_size=2,K_fea=2,HIDDEN_SIZE=20,OUTPUT_SIZE=1,PATH=path,num_epochs=10,LR=0.1,
                          isClassfier=False,MODEL='LSTM',LOSS_NAME='L1Loss',BATCH_SIZE_TRA=4,BATCH_SIZE_VAL=2,BATCH_SIZE_TES=2)

# rnn.load_model_test(path,data,isClassfier=True,isBatchTes=False,Seq=1,K_fea=1)
# print(pred_y)
pred_y.reshape(-1)
data_y.reshape(-1)
plt.plot(pred_y, 'r', label='prediction')
plt.plot(data_y, 'b', label='real')
plt.legend(loc='best')
plt.show()









# if __name__ =='__main__':
#     # load data | construct model | train | save
#     data = np.loadtxt('../data/iris.data',delimiter=',')  # two class
#
#     # print(np.shape(data))
#     # path0 = '/model_save/model_params.pkl'
#     # path1 = '/model_save/model_params.pkl'
#     path = './model_save/model_params.pkl'
#     data_test = data[:,:4]
#
#
#     Flow(data=data,K_fea=4,HIDDEN_SIZE=20,OUTPUT_SIZE=2,PATH=path,num_epochs=10,LR=0.1,isClassfier=True,MODEL='LSTM')
#
#     # load model | predict/test
#     # data_test should only have feature
#     data_x, data_y = load_model_test(path,data_test,isClassfier=True)
#     print(data_y)
#     print(data[:,4])
import numpy as np

a = {'123':[1], 'qwe':'qwe'}

print(a)
# print(isinstance(a,dict))
config_info = {'regParam': '0.0',
                   'model_path': 'hdfs://10.28.0'}

def ddd(regParam=0.1, model_path = 'asd'):
    print(regParam)
    print(model_path)
a = 'False'
print(a)
if a:
    print(0)
else:
    print(1)
b =bool(a)
print(b)
print(type(b))

for i in range(1):
    print(i)

def create_dataset(data_x, data_y, window_size=2):
    """
    目的：处理数据，使用连续的window_size个样本作为特征，最后一个样本的真实值作为标签。
    :param data_x: 所有样本的特征
    :param data_y: 所有样本的真实值
    :param window_size: 窗口大小
    :return: 样本特征 dataX，样本标签 dataY
    """
    dataX, dataY = [], []

    for i in range(len(data_x) - window_size+1):
        a = data_x[i:(i + window_size)]
        # a = a.reshape((window_size,-1))
        # print(np.shape(a))

        dataX.append(a)
        dataY.append(data_y[i + window_size-1])

    # data = np.concatenate((dataX,dataY),axis=0)
    return np.array(dataX), np.array(dataY)

data_x = [[1,2,3],
          [2,3,4],
          [3,4,5]]
data_y = [ 1,
           2,
           3]

data_x, data_y = create_dataset(data_x,data_y,window_size=2)
print(data_x)
print(data_y)
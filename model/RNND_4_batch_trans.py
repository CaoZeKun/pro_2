"""input should be data_x, data_y"""
""" RNN/LSTM 传入 需要batch 匹配，应舍弃最后一个不成批的batch"""
import torch
from torch import nn
import time
import copy
import torch.utils.data as Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# data processing
def data_read_csv(Path_file):
    """
    目的：读取文件
    :param Path_file: :数据文件地址
    :return: 返回dataFrame
    """
    df = pd.read_csv(Path_file, header=None)
    return df

def data_return(isColumnName,df):
    """
    目的：返回列名，以便选取特征和标签 (未考虑特征列名有重复)
    :param isColumnName: 数据文件是否有列名
    :param df: dataFrame
    :return: 文件有列名：返回 列名与下标字典 key_index，列名df.iloc[0, :]，行数 df.shape[0]-1，列数df.shape[1]
             文件无列名：返回 行数 df.shape[0]，列数df.shape[1]
    """

    # 存在列名
    if isColumnName:
        # df = pd.read_csv(Path_file, header=None)
        # key_index = {df.iloc[0,i]: i for i in range(df.shape[1])}
        key_index = {df.iat[0, i]: i for i in range(df.shape[1])}
        return key_index, df.iloc[0, :], (df.shape[0]-1), df.shape[1],

    # 不存在列名
    else:
        # df = pd.read_csv(Path_file, header=None)
        return df.shape[0], df.shape[1]


def receive_read_data(isColumnName,*args):
    if isColumnName:
        key_index = args[0][0]  # key_index
        column_name = args[0][1]  # df.iloc[0, :]
        row_numeber = args[0][2]  # (df.shape[0]-1)
        column_number =args[0][3]  # df.shape[1]

    else:
        row_numeber = args[0][0]  # df.shape[0]
        column_number = args[0][1]  # df.shape[1]
        print(row_numeber,column_number)


def data_processing(dataFrame,isColumnName,*args,**kwargs):
    """
    目的，根据用户选择，选取特征和标签
           ********* Test  Class*********
     case1
    # dataFram = pd.read_csv('../data/iris1.data',header=None,)
    # key_index = {dataFram.iat[0, i]: i for i in range(dataFram.shape[1])}
    # print(key_index)
    # isColumnName = True
    case1 test 1
    # data, k_fea = data_processing(dataFram,isColumnName, key_index = key_index)

    case1 test2
    # args = ['lab']
    # data, k_fea = data_processing(dataFram,isColumnName, args, key_index =key_index)

    case1 test3
    # args = ['lab'],['fea0','fea1']
    # data, k_fea = data_processing(dataFram, isColumnName, args, key_index =key_index)

    case1 test4
    # args = ['lab'],['fea0','fea1'],12
    # data, k_fea = data_processing(dataFram, isColumnName, args, key_index=key_index)

     case2
    # dataFram = pd.read_csv('../data/iris.data',header=None,)
    # isColumnName = False
    case2 test 1
    # data, k_fea = data_processing(dataFram,isColumnName, )

    case2 test2
    # args = [4]
    # data, k_fea = data_processing(dataFram, isColumnName, args)

    case2 test3
    # args = [4],[0,1]
    # data, k_fea = data_processing(dataFram, isColumnName, args)

    case2 test4
    # args = 'lab',['fea0','fea1'],12
    # data, k_fea = data_processing(dataFram, isColumnName, args)

    # data_y, pred_y = rnn.Flow(
    #                         data=data, Seq=1, window_size=1, K_fea=k_fea, HIDDEN_SIZE=20, OUTPUT_SIZE=2, PATH=path,
    #                         num_epochs=20, LR=0.01,isClassfier=True, MODEL='LSTM', BATCH_SIZE_TRA=4, BATCH_SIZE_VAL=1,
    #                         BATCH_SIZE_TES=1)

     ************* Test  Regression *************
    case1
    # data_csv = pd.read_csv('../data/data.csv', usecols=[1])
    # dataFram = pd.read_csv('../data/iris1.data',header=None)
    # key_index = {dataFram.iat[0, i]: i for i in range(dataFram.shape[1])}
    # print(key_index)
    # isColumnName = True
    case1 test 1
    # data, k_fea = data_processing(dataFram,isColumnName, key_index = key_index)

    case1 test2
    # args = ['lab']
    # data, k_fea = data_processing(dataFram,isColumnName, args, key_index =key_index)

    case1 test3
    # args = ['lab'],['fea0','fea1']
    # data, k_fea = data_processing(dataFram, isColumnName, args, key_index =key_index)

    case1 test4
    # args = ['lab'],['fea0','fea1'],12
    # data, k_fea = data_processing(dataFram, isColumnName, args, key_index=key_index)

    case2
    dataFram = pd.read_csv('../data/iris.data', header=None, )
    isColumnName = False
    case2 test 1
    # data, k_fea = data_processing(dataFram, isColumnName, )

    case2 test2
    # args = [4]
    # data, k_fea = data_processing(dataFram, isColumnName, args)

    case2 test3
    # args = [4],[0,1]
    # data, k_fea = data_processing(dataFram, isColumnName, args)

    case2 test4
    # args = 'lab',['fea0','fea1'],12
    # data, k_fea = data_processing(dataFram, isColumnName, args)

    :param dataFrame: 传入之前pandas读文件的DataFram，原因在于文件较大时，重新读入文件，浪费时间。
                       | 也可重新读 dataFrame = pd.readcsv(...)，需重新构建key_index。
    :param isColumnName:  文件是否有列名
    :param args: 若用户未选择，args为空，则默认最后一列为label，其它列为特征。
                 若用户选择只选择某列当标签，应该传入 一个存有标签列名/索引的包含一个元素的list e.g. [2]
                若用户选择某列为标签，某些列为特征，应该传入 一个存有标签列名/索引的list，和一个存有特征列名/索引的列表list
    :param kwargs: 应当传入的是存储列名与下标字典 key_index， 得到的是{key_index ： key_index}
    :return: 元组(特征 np.array(x), 标签 np.array(y)), 特征列数x.shape[1]
              若输入有误，则报错。
    """
    # 存在列名，用户返回应是列名，再寻找索引
    if isColumnName :
        if len(args) == 0:
            x = dataFrame.iloc[1:,:-1]
            y = dataFrame.iloc[1:,-1]
            return (np.array(x), np.array(y)),x.shape[1]
        elif len(args[0]) == 1:
            key_index = kwargs['key_index']
            # print(args[0])
            index_clo = key_index[args[0][0]]
            y = dataFrame.iloc[1:, index_clo]
            x = pd.concat((dataFrame.iloc[1:,:index_clo],dataFrame.iloc[1:,(index_clo+1):]),axis=1)
            return (np.array(x), np.array(y)),x.shape[1]
        elif len(args[0]) == 2:
            key_index = kwargs['key_index']
            index_y = key_index[args[0][0][0]]
            index_x = [key_index[i] for i in args[0][1]]
            y = dataFrame.iloc[1:,index_y]
            x = pd.concat((dataFrame.iloc[1:,i] for i in index_x),axis=1)
            return (np.array(x), np.array(y)),x.shape[1]
        else:
            raise ValueError("The parameters in args should lower 3, not {}".format(len(args[0])))


    # 不存在列名，用户返回应是索引
    else:
        if len(args) == 0:
            x = dataFrame.iloc[:,:-1]
            y = dataFrame.iloc[:,-1]
            return (np.array(x), np.array(y)),x.shape[1]
        elif len(args[0]) == 1:
            index_clo = args[0][0]
            y = dataFrame.iloc[:, index_clo]
            x = pd.concat((dataFrame.iloc[:,:index_clo],dataFrame.iloc[:,(index_clo+1):]),axis=1)
            return (np.array(x), np.array(y)),x.shape[1]
        elif len(args[0]) == 2:
            index_y = args[0][0][0]
            index_x = args[0][1]
            y = dataFrame.iloc[:, index_y]
            x = pd.concat((dataFrame.iloc[:, i] for i in index_x), axis=1)
            return (np.array(x), np.array(y)),x.shape[1]
        else:
            raise ValueError("The parameters in args should lower 3, not {}".format(len(args[0])))


def create_dataset(data_x, data_y, window_size=2):
    """
    目的：处理数据，使用连续的window_size个样本作为特征，最后一个样本的真实值作为标签。
    :param data_x: 所有样本的特征
    :param data_y: 所有样本的真实值
    :param window_size: 窗口大小
    :return: 样本特征 dataX，样本标签 dataY
    """
    dataX, dataY = [], []

    for i in range(len(data_x) - window_size):
        a = data_x[i:(i + window_size)]
        # a = a.reshape((window_size,-1))
        # print(np.shape(a))

        dataX.append(a)
        dataY.append(data_y[i + window_size])

    # data = np.concatenate((dataX,dataY),axis=0)
    return np.array(dataX), np.array(dataY)

# data loader
def load_data_loader(data,k_fea=1,k_train=0.7,k_val=0.2,window_size=2,BATCH_SIZE_TRA=1,BATCH_SIZE_VAL=1,BATCH_SIZE_TES=1,
                     SHUFFLE_BOOL_TRA=False,SHUFFLE_BOOL_VAL=False,SHUFFLE_BOOL_TES=False,NUM_WORKERS_TRA=0,NUM_WORKERS_VAL=0,
                     NUM_WORKERS_TES=0,isClassfier=True,isBatchTes=False,DROP_LAST_TRA=True,DROP_LAST_VAL=True,DROP_LAST_TES=True):
    """
    目的：装载训练/验证/测试数据
    :param data: 数据(元组 （x, y）)
    :param k_fea: 特征的列数，默认：1
    :param k_train: 训练集所占比例，默认：0.7
    :param k_val: 验证集所占比例，默认：0.2
    :param window_size: 窗口大小
    :param BATCH_SIZE_TRA: 训练集批处理量，默认：1
    :param BATCH_SIZE_VAL: 验证集批处理量，默认：1
    :param BATCH_SIZE_TES: 测试集批处理量，默认：1
    :param SHUFFLE_BOOL_TRA: 训练集是否打乱，默认：False （不打乱）
    :param SHUFFLE_BOOL_VAL: 验证集是否打乱，默认：False （不打乱）
    :param SHUFFLE_BOOL_TES: 测试集是否打乱，默认：False （不打乱）
    :param NUM_WORKERS_TRA:  训练集中用于数据加载的子进程数，默认：0
    :param NUM_WORKERS_VAL: 验证集中用于数据加载的子进程数，默认：0
    :param NUM_WORKERS_TES: 测试集中用于数据加载的子进程数，默认：0
    :param DROP_LAST_TRA: 是否丢弃训练集最后一组不够一个批量的样本，默认True
    :param DROP_LAST_VAL: 是否丢弃验证集最后一组不够一个批量的样本，默认True
    :param DROP_LAST_TES: 是否丢弃测试集最后一组不够一个批量的样本，默认True
    :param isClassfier: 是否为分类，默认：True
    :param isBatchTes: 测试集是否使用Batch， 默认: False
    :return: isBatchTes为True 返回 训练数据装载器 train_loader, 验证数据装载器 val_loader, 测试数据装载器 test_loader
             isBatchTes为False 返回 训练数据装载器 train_loader, 验证数据装载器 val_loader, 测试数据元组(x_test,y_test) (测试集特征，测试集真实值)
    """
    # k_fea(特征所在最后一列的索引)
    data_length = len(data[1])
    train_size = int(data_length * k_train)
    val_size = int(data_length * k_val)
    test_size = data_length - val_size
    # data = np.array(data)
    data_x = data[0].astype(float)
    data_y = data[1].astype(float)

    x_train = np.array(data_x[:train_size,])
    y_train = np.array(data_y[:train_size,])
    # print(np.shape(x_train))
    # print(np.shape(y_train))

    x_val = np.array(data_x[train_size:(train_size + val_size), ])
    y_val = np.array(data_y[train_size:(train_size + val_size), ])

    x_test = np.array(data_x[(train_size + val_size):, ])
    y_test = np.array(data_y[(train_size + val_size):, ])

    x_train, y_train = create_dataset(x_train, y_train, window_size=window_size)
    # print(np.shape(x_train))
    x_val, y_val = create_dataset(x_val, y_val, window_size=window_size)
    x_test, y_test = create_dataset(x_test, y_test, window_size=window_size)


    if isClassfier:
        x_train = torch.FloatTensor(x_train)
        y_train = torch.LongTensor(y_train)

        x_val = torch.FloatTensor(x_val)
        y_val = torch.LongTensor(y_val)

        x_test = torch.FloatTensor(x_test)
        y_test = torch.LongTensor(y_test)
        # if LOSSNAME == 'MSELoss':
        #     x_train = torch.FloatTensor(np.array(data[:int(k_train * data_length), :k_fea]))
        #     y_train = torch.FloatTensor(np.array(data[:int(k_train * data_length), k_fea]))
        #
        #     x_val = torch.FloatTensor(np.array(data[int(k_train * data_length):, :k_fea]))
        #     y_val = torch.FloatTensor(np.array(data[int(k_train * data_length):, k_fea]))
        #
        # else:
        #     x_train = torch.FloatTensor(np.array(data[:int(k_train * data_length), :k_fea]))
        #     y_train = torch.LongTensor(np.array(data[:int(k_train * data_length), k_fea]))
        #
        #     x_val = torch.FloatTensor(np.array(data[int(k_train * data_length):, :k_fea]))
        #     y_val = torch.LongTensor(np.array(data[int(k_train * data_length):, k_fea]))
    else:
        x_train = torch.FloatTensor(x_train)
        y_train = torch.FloatTensor(y_train)

        x_val = torch.FloatTensor(x_val)
        y_val = torch.FloatTensor(y_val)

        x_test = torch.FloatTensor(x_test)
        y_test = torch.FloatTensor(y_test)


    train_data = Data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                               batch_size = BATCH_SIZE_TRA,
                                               shuffle = SHUFFLE_BOOL_TRA,
                                               num_workers = NUM_WORKERS_TRA,
                                               drop_last = DROP_LAST_TRA,)
    val_data = Data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(dataset = val_data,
                                             batch_size = BATCH_SIZE_VAL,
                                             shuffle = SHUFFLE_BOOL_VAL,
                                             num_workers = NUM_WORKERS_VAL,
                                             drop_last = DROP_LAST_VAL,)
    if isBatchTes:
        test_data = Data.TensorDataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                             batch_size=BATCH_SIZE_TES,
                                             shuffle=SHUFFLE_BOOL_TES,
                                             num_workers=NUM_WORKERS_TES,
                                             drop_last=DROP_LAST_TES,)
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader, (x_test,y_test)


# model
# RNN
class RNN(nn.Module):
    def __init__(self,INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE,NUM_LAYERS=1,NONLINEARITY='tanh',
                 BIAS_RNN_BOOL=True,BATCH_FIRST=True,DROPOUT_PRO=0,BIDIRECTIONAL_BOOL=False):
        """
        目的：RNN模型初始化
        :param INPUT_SIZE: 数据输入的特征个数
        :param HIDDEN_SIZE: 隐藏层神经元的个数
        :param OUTPUT_SIZE: 输出神经元的个数
        :param NUM_LAYERS: 隐藏层的深度，默认：1
        :param NONLINEARITY: 激活函数，默认：'tanh'
        :param BIAS_RNN_BOOL: 是否有偏置，默认：True
        :param BATCH_FIRST: 调整数据维度，若False (seq_len, batch, input_size), 默认True (batch, seq, input_size)
        :param DROPOUT_PRO: 采用多大概率丢弃，默认：0
        :param BIDIRECTIONAL_BOOL: 是否是双向，默认：False (单向)
        """
        super(RNN,self).__init__()
        self.rnn = nn.RNN(
            input_size = INPUT_SIZE,
            hidden_size = HIDDEN_SIZE,
            num_layers = NUM_LAYERS,
            nonlinearity = NONLINEARITY,  # 'tanh' | 'relu
            bias = BIAS_RNN_BOOL,
            batch_first = BATCH_FIRST,  # data_format (batch, seq, feature)
            dropout = DROPOUT_PRO,
            bidirectional = BIDIRECTIONAL_BOOL,
        )
        self.out = nn.Linear(HIDDEN_SIZE,OUTPUT_SIZE,bias=BIAS_RNN_BOOL)

    # def forward(self,x,h_state):
    def forward(self, x, h):
        """
        目的：前向传播
        :param x: 输入数据维度，BATCH_FIRST 为 True，(batch, seq, input_size)
        :return: 输出数据维度 (batch, output_size)
        """
        # x (batch, seq , feature)
        # h_state (num_layers * num_directions, batch, hidden_size)
        # r_out (batch, seq , num_directions * hidden_size)
        r_out, h_state = self.rnn(x,h)
        # choose r_out at the last time step
        out = self.out(r_out[:,-1,:])

        # return out, h_state
        return out, h_state


# LSTM
class LSTM(nn.Module):
    def __init__(self,INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE,NUM_LAYERS=1,BIAS_LSTM_BOOL=True,
                 BATCH_FIRST=True,DROPOUT_PRO=0,BIDIRECTIONAL_BOOL=False):
        """
        目的：LSTM模型初始化
        :param INPUT_SIZE: 数据输入的特征个数
        :param HIDDEN_SIZE: 隐藏层神经元的个数
        :param OUTPUT_SIZE: 输出神经元的个数
        :param NUM_LAYERS: 隐藏层的深度，默认：1
        :param BIAS_LSTM_BOOL: 是否有偏置，默认：True
        :param BATCH_FIRST: 调整数据维度，若False (seq_len, batch, input_size), 默认True (batch, seq, input_size)
        :param DROPOUT_PRO: 采用多大概率丢弃，默认：0
        :param BIDIRECTIONAL_BOOL: 是否是双向，默认：False (单向)
        """
        super(LSTM,self).__init__()
        self.lstm = nn.LSTM(
            input_size = INPUT_SIZE,
            hidden_size = HIDDEN_SIZE,
            num_layers = NUM_LAYERS,
            bias = BIAS_LSTM_BOOL,
            batch_first = BATCH_FIRST,  # data_format (batch, seq, feature)
            dropout = DROPOUT_PRO,
            bidirectional = BIDIRECTIONAL_BOOL,
        )
        self.out = nn.Linear(HIDDEN_SIZE,OUTPUT_SIZE,bias=BIAS_LSTM_BOOL)


    # def forward(self,x,h_state):
    def forward(self, x, h):
        """
        目的：LSTM模型前向传播
        :param x: 输入数据维度，BATCH_FIRST 为 True，(batch, seq, input_size)
        :return: 输出数据维度 (batch, output_size)
        """
        # x (batch, seq , feature)
        # h_state (num_layers * num_directions, batch, hidden_size)
        # r_out (batch, seq , num_directions * hidden_size)
        r_out, h_n = self.lstm(x,h)
        # print(r_out.size())
        # print(r_out[:,-1,:].size())
        # choose r_out at the last time step
        out = self.out(r_out[:,-1,:])

        # print(out.size())

        # return out
        return out, h_n


def construct_model_opt(INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE,LR=1e-3,OPT = 'Adam',WEIGHT_DECAY=0,
                        LOSS_NAME = 'crossentropy',MODEL = 'RNN', isClassfier=True):
    """
    目的：构建模型，选择优化算法，选择损失函数
    :param INPUT_SIZE: 数据输入的特征个数
    :param HIDDEN_SIZE: 隐藏层神经元的个数
    :param OUTPUT_SIZE: 输出神经元的个数
    :param LR: 学习率，默认：1e-3
    :param OPT: 优化算法，默认：'Adam' ,可选 'Adagrad', 'SGD', 'Adam'
    :param WEIGHT_DECAY: 权重衰减，默认：0
    :param LOSS_NAME: 损失函数名称，默认：'crossentropy' ,可选 'MSELoss', 'crossentropy', 'L1Loss'
    :param MODEL: 模型，默认：'RNN' ,可选'RNN', 'LSTM'
    :param isClassfier:是否为分类，默认：True
    :return: 返回 模型 model, 优化器 optimizer, 损失函数 criterion
    """
    if MODEL not in ['RNN', 'LSTM']:
        raise ValueError(MODEL, "is not an available method.")
    elif MODEL == 'RNN':
        model = RNN(INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE,NUM_LAYERS=1,NONLINEARITY='tanh',
                 BIAS_RNN_BOOL=True,BATCH_FIRST=True,DROPOUT_PRO=0,BIDIRECTIONAL_BOOL=False)
    # else:
    #     model = LSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS=1, NONLINEARITY='tanh',
    #                 BIAS_RNN_BOOL=True, BATCH_FIRST=True, DROPOUT_PRO=0, BIDIRECTIONAL_BOOL=False)
    elif MODEL == 'LSTM':
        model = LSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS=1,BIAS_LSTM_BOOL=True,
                     BATCH_FIRST=True, DROPOUT_PRO=0, BIDIRECTIONAL_BOOL=False)

    if OPT not in ['Adagrad', 'SGD', 'Adam']:
        raise ValueError(OPT, " is not  available.")
    elif OPT == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    elif OPT == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    elif OPT == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if isClassfier:
        if LOSS_NAME == 'crossentropy':
            criterion = nn.CrossEntropyLoss()
            print('crossentropy')
        elif LOSS_NAME == 'MSELoss':
            print("Class ,we recommend crossentropy, not mse")
            criterion = nn.CrossEntropyLoss()
            print('crossentropy')
        else:
            pass
    else:
        if LOSS_NAME == 'MSELoss':
            criterion = nn.MSELoss()
            print('MSELoss')
        else:
            criterion = nn.L1Loss()
            print('L1Loss')

    return model, optimizer, criterion


# train
def train_model(model,train_loader,val_loader,criterion,optimizer,PATH,window_size,num_epochs=1,CUDA_ID="0",
                isClassfier=True,K_fea=1,model_name ='LSTM',BATCH_SIZE_TRA=1,BATCH_SIZE_VAL=1,USE_CUDA=False):
    """
    目的：训练模型
    :param model: 模型
    :param train_loader: 练数据装载器
    :param val_loader: 验证数据装载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param PATH: 模型存储路径
    :param window_size: 窗口大小
    :param num_epochs: 训练迭代次数，默认：1
    :param CUDA_ID: GPU ID号，默认：0
    :param isClassfier: 是否分类，默认：True
    :param K_fea: 特征的列数，默认：1
    :param model_name: RNN， LSTM需要设计不同传入， 默认'RNN'
    :param BATCH_SIZE_TRA: 训练集批处理量，默认：1
    :param BATCH_SIZE_VAL: 验证集批处理量，默认：1
    :param USE_CUDA:是否使用cuda，默认False
    :return: 无返回值，保存最优模型
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:"+CUDA_ID)

    else:
        device = torch.device("cpu")
    if torch.cuda.is_available() and USE_CUDA:
        model.cuda()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict)  # if model is so complex, and metric is not acc,
                                                      # not recommend this, may occupy much memory.
    best_acc = 0.0
    lowest_loss = 100.0
    primary_best_acc = 0.0
    primary_lowest_loss = 100.0


    if model_name == 'RNN':
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            model.train()
            running_loss = 0.0
            running_corrects = 0

            running_loss1 = 0.0
            running_corrects1 = 0

            h_state = None
            for step_0, (train_x, train_y) in enumerate(train_loader):
                # print(train_x.size())
                train_x = train_x.view(-1, window_size, K_fea)
                if torch.cuda.is_available() and USE_CUDA:
                    train_x = train_x.cuda()
                    train_y = train_y.cuda()
                # print(train_x.size())
                # print(train_y.size())
                # output_tra, h_state= model(train_x,h_state) #  output
                output_tra, h_state = model(train_x, h_state)  # output
                # print(output_tra.size())
                h_state = h_state.data
                loss_tra = criterion(output_tra, train_y)
                optimizer.zero_grad()  # clear gradients for this training step
                loss_tra.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                _, tra_preds = torch.max(output_tra, 1)
                running_loss += loss_tra.item() * train_x.size(0)
                if isClassfier:
                    running_corrects += torch.sum(tra_preds == train_y.data)

            epoch_tra_loss = running_loss / len(train_loader.dataset)
            if isClassfier:
                epoch_tra_acc = running_corrects.double() / len(train_loader.dataset)
                print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_tra_loss, epoch_tra_acc))
            else:
                print('Train Loss: {:.4f} '.format(epoch_tra_loss))


            h_state1 = None
            model.eval()
            for step_1, (val_x, val_y) in enumerate(val_loader):

                val_x = val_x.view(-1, window_size, K_fea)
                if torch.cuda.is_available() and USE_CUDA:
                    val_x = val_x.cuda()
                    val_y = val_y.cuda()
                # output_val = model(val_x,h_state) #  output
                output_val, _ = model(val_x, h_state1)  # output
                loss = criterion(output_val, val_y)  # cross entropy loss

                _, val_preds = torch.max(output_val, 1)
                running_loss1 += loss.item() * val_x.size(0)
                if isClassfier:

                    running_corrects1 += torch.sum(val_preds == val_y.data)

            epoch_val_loss = running_loss1 / len(val_loader.dataset)
            if isClassfier:
                epoch_val_acc = running_corrects1.double() / len(val_loader.dataset)
                print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_val_loss, epoch_val_acc))
                # deep copy the model
                if epoch_val_acc > best_acc:
                    best_acc = epoch_val_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                val_acc_history.append(epoch_val_acc)
            else:
                if epoch_val_loss < lowest_loss:
                    lowest_loss = epoch_val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                print('Val Loss: {:.4f} '.format(epoch_val_loss))

    else:
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            model.train()
            running_loss = 0.0
            running_corrects = 0
            running_loss1 = 0.0
            running_corrects1 = 0

            h_n = None
            for step_0, (train_x, train_y) in enumerate(train_loader):
                # print(train_x.size())
                train_x = train_x.view(-1, window_size, K_fea)
                if torch.cuda.is_available() and USE_CUDA:
                    train_x = train_x.cuda()
                    train_y = train_y.cuda()

                # print(train_x.size())
                # print(train_y.size())
                # output_tra, h_state= model(train_x,h_state) #  output
                output_tra, h_n = model(train_x, h_n)  # output
                # print(output_tra.size())
                h_state = h_n[0].data
                h_c = h_n[1].data
                h_n = (h_state,h_c)
                loss_tra = criterion(output_tra, train_y)
                optimizer.zero_grad()  # clear gradients for this training step
                loss_tra.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                _, tra_preds = torch.max(output_tra, 1)
                running_loss += loss_tra.item() * train_x.size(0)
                if isClassfier:
                    running_corrects += torch.sum(tra_preds == train_y.data)

            epoch_tra_loss = running_loss / len(train_loader.dataset)
            if isClassfier:
                epoch_tra_acc = running_corrects.double() / len(train_loader.dataset)
                print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_tra_loss, epoch_tra_acc))
            else:
                print('Train Loss: {:.4f} '.format(epoch_tra_loss))



            h_state1 = None
            model.eval()
            for step_1, (val_x, val_y) in enumerate(val_loader):

                val_x = val_x.view(-1, window_size, K_fea)
                # output_val = model(val_x,h_state) #  output
                if torch.cuda.is_available() and USE_CUDA:
                    val_x = val_x.cuda()
                    val_y = val_y.cuda()
                output_val, _ = model(val_x, h_state1)  # output
                loss = criterion(output_val, val_y)  # cross entropy loss

                _, val_preds = torch.max(output_val, 1)
                running_loss1 += loss.item() * val_x.size(0)
                if isClassfier:
                    running_corrects1 += torch.sum(val_preds == val_y.data)
            # print(running_corrects1)
            epoch_val_loss = running_loss1 / len(val_loader.dataset)
            # deep copy the model
            if isClassfier:
                epoch_val_acc = running_corrects1.double() / len(val_loader.dataset)
                print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_val_loss, epoch_val_acc))
                if epoch_val_acc > best_acc:
                    best_acc = epoch_val_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                val_acc_history.append(epoch_val_acc)
            else:
                if epoch_val_loss < lowest_loss:
                    lowest_loss = epoch_val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                print('Val Loss: {:.4f} '.format(epoch_val_loss))



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    if isClassfier:
        print('Best val Acc: {:4f}'.format(best_acc))
    else:
        print('Lowest loss: {:4f}'.format(lowest_loss))

    if best_acc == primary_best_acc or lowest_loss == primary_lowest_loss:
        torch.save(model, PATH)
    else:
        # load best model weights
    # print(best_model_wts)
        model.load_state_dict(best_model_wts)
        # torch.save(the_model, PATH)
        torch.save(model, PATH)
    # return model, val_acc_history


# test
def Flow(data,HIDDEN_SIZE, OUTPUT_SIZE, PATH, Seq=1,window_size=2, K_fea=1,k_train=0.7,k_val=0.2, num_epochs=1, LR=1e-3,LOSS_NAME = 'crossentropy',
         CUDA_ID="0", isClassfier=True, MODEL='RNN',isBatchTes=False,BATCH_SIZE_TRA=1,BATCH_SIZE_VAL=1,BATCH_SIZE_TES=1,USE_CUDA=False):
    """
    目的：整体流程: 数据装载 -> 模型构建 -> 模型训练(保存)
    若是分类，OUTPUT_SIZE应该与标签Label类别数一致；
    若是回归，OUTPUT_SIZE应该为1
    :param data: 数据
    :param HIDDEN_SIZE: 隐藏层神经元的个数
    :param OUTPUT_SIZE: 输出神经元的个数
    :param PATH:模型存储路径
    :param Seq: 时间序列数，默认：1
    :param window_size: 窗口大小
    :param K_fea: 特征的列数，默认：1
    :param k_train: 训练集所占比例，默认：0.7
    :param k_val: 验证集所占比例，默认：0.2
    :param num_epochs:训练迭代次数，默认：1
    :param LR:学习率，默认：1e-3
    :param LOSS_NAME:损失函数名称，默认：'crossentropy'
    :param CUDA_ID:GPU ID号，默认：0
    :param isClassfier:是否分类，默认：True
    :param MODEL:模型, 默认:'RNN'
    :param isBatchTes: 测试集是否使用Batch， 默认: False
    :param BATCH_SIZE_TRA: 训练集批处理量，默认：1
    :param BATCH_SIZE_VAL: 验证集批处理量，默认：1
    :param BATCH_SIZE_TES: 测试集批处理量，默认：1
    :param USE_CUDA:是否使用cuda，默认False
    :return:无返回值，train_model内保存最优模型
    """
    train_loader, val_loader, test_loader = load_data_loader(data, K_fea, window_size=window_size,k_train=k_train,k_val=k_val, BATCH_SIZE_TRA=BATCH_SIZE_TRA, BATCH_SIZE_VAL=BATCH_SIZE_VAL,BATCH_SIZE_TES=BATCH_SIZE_TES, SHUFFLE_BOOL_TRA=False,
              SHUFFLE_BOOL_VAL=False, SHUFFLE_BOOL_TES=True,NUM_WORKERS_TRA=0, NUM_WORKERS_VAL=0, NUM_WORKERS_TES=0,isClassfier=isClassfier,isBatchTes=isBatchTes)
    model, optimizer, criterion = construct_model_opt(K_fea, HIDDEN_SIZE, OUTPUT_SIZE, LR=LR, OPT='Adam', WEIGHT_DECAY=0,
                        LOSS_NAME=LOSS_NAME, MODEL=MODEL,isClassfier=isClassfier)
    train_model(model, train_loader, val_loader, criterion, optimizer,PATH,window_size,num_epochs,CUDA_ID,isClassfier,K_fea=K_fea,model_name=MODEL,BATCH_SIZE_TRA=BATCH_SIZE_TRA,BATCH_SIZE_VAL=BATCH_SIZE_VAL,USE_CUDA=USE_CUDA)

    """For test"""
    if isBatchTes:
        data_y, pred_y = load_model_test(PATH,test_loader,isClassfier=isClassfier,isBatchTes=isBatchTes,Seq=Seq,K_fea=K_fea,BATCH_SIZE_TES=BATCH_SIZE_TES,USE_CUDA=USE_CUDA)
    else:
        # print(type(test_loader))
        data_y, pred_y = load_model_test(PATH, test_loader, isClassfier=isClassfier, isBatchTes=isBatchTes,Seq=Seq,K_fea=K_fea,USE_CUDA=USE_CUDA)

    return data_y, pred_y


# save whole model
def load_model_test(PATH,data,isClassfier=True,isBatchTes=False,Seq=1,K_fea=1,CUDA_ID="0",BATCH_SIZE_TES=1,USE_CUDA=False):
    """
    目的：加载训练好的模型，进行测试/预测
    :param PATH:保存的模型路径
    :param data:测试数据
    :param isClassfier:是否为分类，默认True
    :param isBatchTes: 测试集是否使用Batch， 默认: False
    :param Seq: 时间序列数，默认：1
    :param K_fea: 特征的列数，默认：1
    :param CUDA_ID: GPU ID号，默认：0
    :param BATCH_SIZE_TES: 测试集批处理量，默认：1
    :param USE_CUDA:是否使用cuda，默认False
    :return:测试数据的真实值 data_y 测试数据的预测结果 pred_y
    """
    device = torch.device(("cuda:"+CUDA_ID) if torch.cuda.is_available() else "cpu")
    # Model class must be defined somewhere
    model = torch.load(PATH)
    model.eval()

    if isBatchTes:
        pred_y = torch.Tensor([])
        if isClassfier:
            data_y = torch.LongTensor([])
        else:
            data_y = torch.Tensor([])
        # data_x = torch.Tensor([])
        h = None
        for step_0,(test_x, test_y) in enumerate(data):
            test_x = test_x.view(-1,Seq,K_fea)
            if torch.cuda.is_available() and USE_CUDA:
                test_x = test_x.cuda()
            output, _ = model(test_x,h)
            # print(output.size())
            pred_y = torch.cat((pred_y,output),0)
            data_y = torch.cat((data_y, test_y), 0)
            # data_x = torch.cat((data_x, test_x), 0)

    else:
        # 1. simple, not many samples
        data_x = torch.Tensor(np.array(data[0]))
        data_y = data[1]

        data_x = data_x.view(-1,Seq,K_fea)
        # print(data_x.size())
        if torch.cuda.is_available() and USE_CUDA :
            data_x = data_x.cuda()
        h = None
        pred_y, _ = model(data_x, h)

    if isClassfier:
        _, pred_y = torch.max(pred_y, 1)
    else:
        # pred_y, _ = torch.max(pred_y, 1)
        pass

    return data_y.cpu().detach().numpy(), pred_y.cpu().detach().numpy()


# save parameters
# def load_param_test(model,TheModelClass,PATH,):
#     torch.save(model.state_dict(), PATH)
#     ###
#     model = TheModelClass(*args, **kwargs)
#     model.load_state_dict(torch.load(PATH))
#     model.eval()





if __name__ =='__main__':
    path = './model_save/model_params.pkl'

    """         *** Test  Class***         """
    """ case1 """
    dataFram = pd.read_csv('../data/iris1.data',header=None,)
    key_index = {dataFram.iat[0, i]: i for i in range(dataFram.shape[1])}
    isColumnName = True
    """case1 test 1"""
    data, k_fea = data_processing(dataFram,isColumnName, key_index = key_index)

    """case1 test2"""
    # args = ['lab']
    # data, k_fea = data_processing(dataFram,isColumnName, args, key_index =key_index)

    """case1 test3"""
    # args = ['lab'],['fea0','fea1']
    # data, k_fea = data_processing(dataFram, isColumnName, args, key_index =key_index)

    """case1 test4"""
    # args = ['lab'],['fea0','fea1'],12
    # data, k_fea = data_processing(dataFram, isColumnName, args, key_index=key_index)

    """ case2 """
    # dataFram = pd.read_csv('../data/iris.data',header=None,)
    # isColumnName = False
    # """case2 test 1"""
    # data, k_fea = data_processing(dataFram,isColumnName, )

    """case2 test2"""
    # args = [4]
    # data, k_fea = data_processing(dataFram, isColumnName, args)

    """case2 test3"""
    # args = [4],[0,1]
    # data, k_fea = data_processing(dataFram, isColumnName, args)

    """case2 test4"""
    # args = 'lab',['fea0','fea1'],12
    # data, k_fea = data_processing(dataFram, isColumnName, args)

    data_y, pred_y = Flow(
                            data=data, Seq=1, window_size=1, K_fea=k_fea, HIDDEN_SIZE=20, OUTPUT_SIZE=2, PATH=path,
                            num_epochs=20, LR=0.01,isClassfier=True, MODEL='RNN', BATCH_SIZE_TRA=4, BATCH_SIZE_VAL=1,
                            BATCH_SIZE_TES=1,isBatchTes=False,USE_CUDA=False)

    """       ************* Test  Regression *************        """
    """ case1 """
    # data_csv = pd.read_csv('../data/data.csv', usecols=[1])
    # dataFram = data_read_csv('../data/iris1.data')
    # key_index = {dataFram.iat[0, i]: i for i in range(dataFram.shape[1])}
    # print(key_index)
    # isColumnName = True
    """case1 test 1"""
    # data, k_fea = data_processing(dataFram,isColumnName, key_index = key_index)

    """case1 test2"""
    # args = ['lab']
    # data, k_fea = data_processing(dataFram,isColumnName, args, key_index =key_index)

    """case1 test3"""
    # args = ['lab'],['fea0','fea1']
    # data, k_fea = data_processing(dataFram, isColumnName, args, key_index =key_index)

    """case1 test4"""
    # args = ['lab'],['fea0','fea1'],12
    # data, k_fea = data_processing(dataFram, isColumnName, args, key_index=key_index)

    """ case2 """
    # dataFram = data_read_csv('../data/iris.data')
    # isColumnName = False
    """case2 test 1"""
    # data, k_fea = data_processing(dataFram, isColumnName, )

    """case2 test2"""
    # args = [4]
    # data, k_fea = data_processing(dataFram, isColumnName, args)

    """case2 test3"""
    # args = [4],[0,1]
    # data, k_fea = data_processing(dataFram, isColumnName, args)

    """case2 test4"""
    # args = 'lab',['fea0','fea1'],12
    # data, k_fea = data_processing(dataFram, isColumnName, args)

    # data_y, pred_y = Flow(data=data,Seq=1,window_size=2, K_fea=k_fea,HIDDEN_SIZE=20,OUTPUT_SIZE=1,PATH=path,num_epochs=10,LR=0.1,
    #                       isClassfier=False,MODEL='LSTM',BATCH_SIZE_TRA=4,BATCH_SIZE_VAL=1,BATCH_SIZE_TES=1)

    """ Test data_read"""

    # df = data_read_csv('../data/iris1.data')
    # isColumnName = True
    # res = data_return(isColumnName,df)
    # receive_read_data(isColumnName,res)

    # df = data_read_csv('../data/iris.data')
    # isColumnName = False
    # res = data_return(isColumnName,df)
    # receive_read_data(False, res)







"""input should be data_x, data_y"""
import torch
from torch import nn
import time
import copy
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt


# data processing
def data_processing():
    pass


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
def load_data_loader1(data,k_fea=1,k_train=0.7,k_val=0.2,window_size=2,BATCH_SIZE_TRA=1,BATCH_SIZE_VAL=1,BATCH_SIZE_TES=1,
                     SHUFFLE_BOOL_TRA=False,SHUFFLE_BOOL_VAL=False,SHUFFLE_BOOL_TES=False,NUM_WORKERS_TRA=0,NUM_WORKERS_VAL=0,
                     NUM_WORKERS_TES=0,isClassfier=True,isBatchTes=False,DROP_LAST_TRA=False,DROP_LAST_VAL=False,DROP_LAST_TES=False):
    """
    目的：装载训练/验证/测试数据
    :param data: 数据
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
    :param DROP_LAST_TRA: 是否丢弃训练集最后一组不够一个批量的样本，默认False
    :param DROP_LAST_VAL: 是否丢弃验证集最后一组不够一个批量的样本，默认False
    :param DROP_LAST_TES: 是否丢弃测试集最后一组不够一个批量的样本，默认False
    :param isClassfier: 是否为分类，默认：True
    :param isBatchTes: 测试集是否使用Batch， 默认: False
    :return: isBatchTes为True 返回 训练数据装载器 train_loader, 验证数据装载器 val_loader, 测试数据装载器 test_loader
             isBatchTes为False 返回 训练数据装载器 train_loader, 验证数据装载器 val_loader, 测试数据元组(x_test,y_test) (测试集特征，测试集真实值)
    """
    # k_fea(特征所在最后一列的索引)
    data_length = len(data)
    train_size = int(data_length * k_train)
    val_size = int(data_length * k_val)
    test_size = data_length - val_size
    # data = np.array(data)

    x_train = np.array(data[:train_size, :k_fea])
    y_train = np.array(data[:train_size, k_fea])

    x_val = np.array(data[train_size:(train_size + val_size), :k_fea])
    y_val = np.array(data[train_size:(train_size + val_size), k_fea])

    x_test = np.array(data[(train_size + val_size):, :k_fea])
    y_test = np.array(data[(train_size + val_size):, k_fea])

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

def load_data_loader(data,k_fea=1,k_train=0.7,k_val=0.2,window_size=2,BATCH_SIZE_TRA=1,BATCH_SIZE_VAL=1,BATCH_SIZE_TES=1,
                     SHUFFLE_BOOL_TRA=False,SHUFFLE_BOOL_VAL=False,SHUFFLE_BOOL_TES=False,NUM_WORKERS_TRA=0,NUM_WORKERS_VAL=0,
                     NUM_WORKERS_TES=0,isClassfier=True,isBatchTes=False,DROP_LAST_TRA=False,DROP_LAST_VAL=False,DROP_LAST_TES=False):
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
    :param DROP_LAST_TRA: 是否丢弃训练集最后一组不够一个批量的样本，默认False
    :param DROP_LAST_VAL: 是否丢弃验证集最后一组不够一个批量的样本，默认False
    :param DROP_LAST_TES: 是否丢弃测试集最后一组不够一个批量的样本，默认False
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
    data_x = data[0]
    data_y = data[1]

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
    def forward(self, x):
        """
        目的：前向传播
        :param x: 输入数据维度，BATCH_FIRST 为 True，(batch, seq, input_size)
        :return: 输出数据维度 (batch, output_size)
        """
        # x (batch, seq , feature)
        # h_state (num_layers * num_directions, batch, hidden_size)
        # r_out (batch, seq , num_directions * hidden_size)
        r_out, h_state = self.rnn(x)
        # choose r_out at the last time step
        out = self.out(r_out[:,-1,:])

        # return out, h_state
        return out


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
    def forward(self, x):
        """
        目的：LSTM模型前向传播
        :param x: 输入数据维度，BATCH_FIRST 为 True，(batch, seq, input_size)
        :return: 输出数据维度 (batch, output_size)
        """
        # x (batch, seq , feature)
        # h_state (num_layers * num_directions, batch, hidden_size)
        # r_out (batch, seq , num_directions * hidden_size)
        r_out, (h_n, h_c) = self.lstm(x)
        # print(r_out.size())
        # print(r_out[:,-1,:].size())
        # choose r_out at the last time step
        out = self.out(r_out[:,-1,:])

        # print(out.size())

        # return out
        return out



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

    if MODEL == 'RNN':
        model = RNN(INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE,NUM_LAYERS=1,NONLINEARITY='tanh',
                 BIAS_RNN_BOOL=True,BATCH_FIRST=True,DROPOUT_PRO=0,BIDIRECTIONAL_BOOL=False)
    # else:
    #     model = LSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS=1, NONLINEARITY='tanh',
    #                 BIAS_RNN_BOOL=True, BATCH_FIRST=True, DROPOUT_PRO=0, BIDIRECTIONAL_BOOL=False)
    if MODEL == 'LSTM':
        model = LSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS=1,BIAS_LSTM_BOOL=True,
                     BATCH_FIRST=True, DROPOUT_PRO=0, BIDIRECTIONAL_BOOL=False)



    if OPT == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(),lr = LR, weight_decay=WEIGHT_DECAY)
    elif OPT == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    elif OPT == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
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
                isClassfier=True,K_fea=1,BATCH_SIZE_TRA=1,BATCH_SIZE_VAL=1,):
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
    :param BATCH_SIZE_TRA: 训练集批处理量，默认：1
    :param BATCH_SIZE_VAL: 验证集批处理量，默认：1
    :return: 无返回值，保存最优模型
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:"+CUDA_ID)

    else:
        device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict)  # if model is so complex, and metric is not acc,
                                                      # not recommend this, may occupy much memory.
    best_acc = 0.0
    lowest_loss = 100.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        # h_state = 0
        for step_0, (train_x, train_y) in enumerate(train_loader):
            # print(train_x.size())
            train_x = train_x.view(-1,window_size,K_fea)
            # print(train_x.size())
            # print(train_y.size())
            # output_tra, h_state= model(train_x,h_state) #  output
            output_tra = model(train_x)  # output
            # print(output_tra.size())
            # h_state = h_state.data
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

        running_loss = 0.0
        running_corrects = 0
        h_state = 0
        model.eval()
        for step_1, (val_x, val_y) in enumerate(val_loader):

            val_x = val_x.view(-1, window_size, K_fea)
            # output_val = model(val_x,h_state) #  output
            output_val = model(val_x)  # output
            loss = criterion(output_val, val_y)  # cross entropy loss

            _, val_preds = torch.max(output_val, 1)
            running_loss += loss.item() * val_x.size(0)
            if isClassfier:
                running_corrects += torch.sum(val_preds == val_y.data)

        epoch_val_loss = running_loss / len(val_loader.dataset)
        if isClassfier:
            epoch_val_acc = running_corrects.double() / len(val_loader.dataset)
            print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_val_loss, epoch_val_acc))
            # deep copy the model
            if  epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            val_acc_history.append(epoch_val_acc)
        else:
            if  epoch_val_loss < lowest_loss:
                lowest_loss = epoch_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            print('Val Loss: {:.4f} '.format(epoch_val_loss))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    if isClassfier:
        print('Best val Acc: {:4f}'.format(best_acc))
    else:
        print('Lowest loss: {:4f}'.format(lowest_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # torch.save(the_model, PATH)
    torch.save(model, PATH)
    # return model, val_acc_history


# test
def Flow(data,HIDDEN_SIZE, OUTPUT_SIZE, PATH, Seq=1,window_size=2, K_fea=1,k_train=0.7,k_val=0.2, num_epochs=1, LR=1e-3,LOSS_NAME = 'crossentropy',
         CUDA_ID="0", isClassfier=True, MODEL='RNN',isBatchTes=False,BATCH_SIZE_TRA=1,BATCH_SIZE_VAL=1,BATCH_SIZE_TES=1):
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
    :return:无返回值，train_model内保存最优模型
    """
    train_loader, val_loader, test_loader = load_data_loader(data, K_fea, window_size=window_size,k_train=k_train,k_val=k_val, BATCH_SIZE_TRA=BATCH_SIZE_TRA, BATCH_SIZE_VAL=BATCH_SIZE_VAL,BATCH_SIZE_TES=BATCH_SIZE_TES, SHUFFLE_BOOL_TRA=False,
              SHUFFLE_BOOL_VAL=False, SHUFFLE_BOOL_TES=True,NUM_WORKERS_TRA=0, NUM_WORKERS_VAL=0, NUM_WORKERS_TES=0,isClassfier=isClassfier,isBatchTes=False)
    model, optimizer, criterion = construct_model_opt(K_fea, HIDDEN_SIZE, OUTPUT_SIZE, LR=LR, OPT='Adam', WEIGHT_DECAY=0,
                        LOSS_NAME=LOSS_NAME, MODEL=MODEL,isClassfier=isClassfier)
    train_model(model, train_loader, val_loader, criterion, optimizer,PATH,window_size,num_epochs,CUDA_ID,isClassfier,K_fea=K_fea,BATCH_SIZE_TRA=BATCH_SIZE_TRA,BATCH_SIZE_VAL=BATCH_SIZE_VAL,)

    """For test"""
    if isBatchTes:
        data_y, pred_y = load_model_test(PATH,test_loader,isClassfier=isClassfier,isBatchTes=isBatchTes,Seq=Seq,K_fea=K_fea,BATCH_SIZE_TES=BATCH_SIZE_TES)
    else:
        # print(type(test_loader))
        data_y, pred_y = load_model_test(PATH, test_loader, isClassfier=isClassfier, isBatchTes=isBatchTes,Seq=Seq,K_fea=K_fea)

    return data_y, pred_y


# save whole model
def load_model_test(PATH,data,isClassfier=True,isBatchTes=False,Seq=1,K_fea=1,CUDA_ID="0",BATCH_SIZE_TES=1):
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
    :return:测试数据的真实值 data_y 测试数据的预测结果 pred_y
    """
    USE_CUDA = torch.cuda.is_available()
    device = torch.device(("cuda:"+CUDA_ID) if USE_CUDA else "cpu")
    # Model class must be defined somewhere
    model = torch.load(PATH)
    model.eval()

    if isBatchTes:
        pred_y = torch.Tensor([])
        data_y = torch.Tensor([])
        # data_x = torch.Tensor([])
        for step_0,(test_x, test_y) in enumerate(data):
            test_x = test_x.view(-1,Seq,K_fea)
            output = model(test_x)
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
        pred_y = model(data_x)

    if isClassfier:
        _, pred_y = torch.max(pred_y, 1)
    else:
        # pred_y, _ = torch.max(pred_y, 1)
        pass

    return data_y.detach().numpy(), pred_y.detach().numpy()


# save parameters
# def load_param_test(model,TheModelClass,PATH,):
#     torch.save(model.state_dict(), PATH)
#     ###
#     model = TheModelClass(*args, **kwargs)
#     model.load_state_dict(torch.load(PATH))
#     model.eval()





if __name__ =='__main__':
    # load data | construct model | train | save
    data = np.loadtxt('../data/iris.data',delimiter=',')  # two class
    # print(np.shape(data))
    # path0 = '/model_save/model_params.pkl'
    # path1 = '/model_save/model_params.pkl'
    path = './model_save/model_params.pkl'
    data_test = data[:,:4]

    data_y, pred_y = Flow(data=data,Seq=1,window_size=1, K_fea=4,HIDDEN_SIZE=20,OUTPUT_SIZE=2,PATH=path,num_epochs=10,LR=0.1,
                          isClassfier=True,MODEL='LSTM',BATCH_SIZE_TRA=4,BATCH_SIZE_VAL=1,BATCH_SIZE_TES=1)

    """load model | predict/test"""
    # data_test should only have feature
    # data_y, pred_y = load_model_test(path,data_test,isClassfier=True,Seq=1,K_fea=4)
    print(pred_y)
    print(data[:,4])






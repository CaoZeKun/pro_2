import torch
from torch import nn
import rnn


# model
# RNN
class RNNRegression(nn.Module):
    def __init__(self ,INPUT_SIZE ,HIDDEN_SIZE ,OUTPUT_SIZE ,NUM_LAYERS=1 ,NONLINEARITY='tanh',
                 BIAS_RNN_BOOL=True ,BATCH_FIRST=True ,DROPOUT_PRO=0 ,BIDIRECTIONAL_BOOL=False):
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
        super(RNNRegression,self).__init__()
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
        self.out = nn.Linear(HIDDEN_SIZE ,OUTPUT_SIZE ,bias=BIAS_RNN_BOOL)

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
        outs = self.out(r_out)

        return outs


# LSTM
class LSTMRegression(nn.Module):
    def __init__(self ,INPUT_SIZE ,HIDDEN_SIZE ,OUTPUT_SIZE ,NUM_LAYERS=1 ,BIAS_LSTM_BOOL=True,
                 BATCH_FIRST=True ,DROPOUT_PRO=0 ,BIDIRECTIONAL_BOOL=False):
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
        super(LSTMRegression ,self).__init__()
        self.lstm = nn.LSTM(
            input_size = INPUT_SIZE,
            hidden_size = HIDDEN_SIZE,
            num_layers = NUM_LAYERS,
            bias = BIAS_LSTM_BOOL,
            batch_first = BATCH_FIRST,  # data_format (batch, seq, feature)
            dropout = DROPOUT_PRO,
            bidirectional = BIDIRECTIONAL_BOOL,
        )
        self.out = nn.Linear(HIDDEN_SIZE ,OUTPUT_SIZE ,bias=BIAS_LSTM_BOOL)

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
        print(r_out.size())
        print(r_out[: ,-1 ,:].size())
        out = self.out(r_out)

        return out

if __name__ =='__main__':

"""
Create on Fri Jan 25 2019

@author Yangkun Cao
"""

import RNN1DBatchNoTrans as RNNnoTRA
import numpy as np
import pandas as pd


class DataSelect():
    """
    标签选取，特征选取
    """

    name = "DataSelect"

    # ##@staticmethod
    def data_processing(self, dataFrame, isColumnName, *args, **kwargs):
        """
        目的，根据用户选择列名或下标，选取特征和标签

       :param dataFrame: 传入之前pandas已读取文件的DataFram，原因在于文件较大时，重新读入文件，浪费时间。
                           | 也可重新读 dataFrame = pd.readcsv(...)，需重新构建key_index。
        :param isColumnName:  文件是否有列名
        :param args: 若用户未选择，args为空，则默认最后一列为label，其它列为特征。
                     若用户选择只选择某列当标签(默认余下列为特征)，应该传入 一个存有标签列名/索引的包含一个元素的list e.g. [2]
                    若用户选择某列为标签，某些列为特征，应该传入 一个存有标签列名/索引的list，和一个存有特征列名/索引的列表list
        :param kwargs: 应当传入的是存储列名与下标字典 key_index， 得到的是{key_index ： key_index}
        :return: 元组(特征 np.array(x), 标签 np.array(y)), 特征列数x.shape[1]
                  若输入有误，则报错。
        """
        # 存在列名，用户返回应是列名，再寻找索引
        if isColumnName:
            if len(args) == 0:
                x = dataFrame.iloc[1:, :-1]
                y = dataFrame.iloc[1:, -1]
                return (np.array(x), np.array(y)), x.shape[1]
            elif len(args[0]) == 1:
                key_index = kwargs['key_index']
                # print(args[0])
                index_clo = key_index[args[0][0]]
                y = dataFrame.iloc[1:, index_clo]
                x = pd.concat((dataFrame.iloc[1:, :index_clo], dataFrame.iloc[1:, (index_clo + 1):]), axis=1)
                return (np.array(x), np.array(y)), x.shape[1]
            elif len(args[0]) == 2:
                key_index = kwargs['key_index']
                index_y = key_index[args[0][0][0]]
                index_x = [key_index[i] for i in args[0][1]]
                y = dataFrame.iloc[1:, index_y]
                x = pd.concat((dataFrame.iloc[1:, i] for i in index_x), axis=1)
                return (np.array(x), np.array(y)), x.shape[1]
            else:
                assert len(args[0]) < 3, 'args,传入参数，应该小于3'

        # 不存在列名，用户返回应是索引
        else:
            if len(args) == 0:
                x = dataFrame.iloc[:, :-1]
                y = dataFrame.iloc[:, -1]
                return (np.array(x), np.array(y)), x.shape[1]
            elif len(args[0]) == 1:
                index_clo = args[0][0]
                y = dataFrame.iloc[:, index_clo]
                x = pd.concat((dataFrame.iloc[:, :index_clo], dataFrame.iloc[:, (index_clo + 1):]), axis=1)
                return (np.array(x), np.array(y)), x.shape[1]
            elif len(args[0]) == 2:
                index_y = args[0][0][0]
                index_x = args[0][1]
                y = dataFrame.iloc[:, index_y]
                x = pd.concat((dataFrame.iloc[:, i] for i in index_x), axis=1)
                return (np.array(x), np.array(y)), x.shape[1]
            else:
                assert len(args[0]) < 3, 'args,传入参数，应该小于3'




if __name__ =='__main__':
    path = './model_save/model_params.pkl'

    """         *** Test  Class***         """
    """ case1 """
    dataFram = pd.read_csv('iris1.data',header=0,)
    print(dataFram.head(10))
    print(dataFram.columns)
    key_index = {dataFram.iat[0, i]: i for i in range(dataFram.shape[1])}
    isColumnName = True

    """case1 test 1"""
    data, k_fea = DataSelect().data_processing(dataFram,isColumnName, key_index = key_index)

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

    # data_y, pred_y = Flow(
    #                         data=data, Seq=1, window_size=1, K_fea=k_fea, HIDDEN_SIZE=20, OUTPUT_SIZE=2, PATH=path,
    #                         num_epochs=10, LR=0.1,isClassfier=True, MODEL='RNN', BATCH_SIZE_TRA=4, BATCH_SIZE_VAL=1,
    #                         BATCH_SIZE_TES=1,USE_CUDA=False,isBatchTes=False)

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























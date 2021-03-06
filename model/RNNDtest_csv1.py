import pandas as pd
import numpy as np

# df = pd.read_csv('../data/iris.data',header=None,)
# print(df.loc[0])
# print(df.head(1))
# df1 = df.head(1)
# print(df1[1])
# key_index = {df.iat[0,i]: i for i in range(df.shape[1])}
# print(key_index)
# print(df.iloc[1][0])
# key_index ={ df.iloc[0][i]:i  for i in range(df.shape[1]) }
# index_clo=2
# print(df.iloc[1:3,:index_clo])
# print(df.iloc[1:3,(index_clo+1):])
# x = pd.concat((df.iloc[1:3,:index_clo],df.iloc[1:3,(index_clo+1):]),axis=1)
# x = pd.concat((df.iloc[:2,i] for i in [0,2,4]),axis=1)
#
# print(x)
# print(df.shape[0], df.shape[1])
# print(key_index['fea3'])
# print(df.iloc[0,:])
# print(df.iloc[0,:][2].index)
# print(df.columns.values)
# print(df.index.values)
# print(df.columns.values.tolist())
# print(df.loc[df =='xiajjz_ffz'])
# print(df[df['exc_current_mv'] == True].index.tolist())

# df = np.array(df)
# print(np.shape(df))
# print(df[:10,:10])
# assert
# assert 2+2==2*2,'aa'
# assert 2+2==2*1,'aa'

import RNN1D_2 as rnn
# df = pd.read_csv('../data/hd_ffz_mean_tmp.csv', header=None)

def data_read_csv(isColumnName,Path_file):
    """
    目的：返回列名，以便选取特征和标签 (未考虑特征列名有重复)
    :param isColumnName: 数据文件是否有列名
    :param Path_file: 数据文件地址
    :return: 文件有列名：返回 列名与下标字典 key_index，列名df.iloc[0, :]，行数 df.shape[0]-1，列数df.shape[1]
             文件无列名：返回 行数 df.shape[0]，列数df.shape[1]
    """

    # 存在列名
    if isColumnName:
        df = pd.read_csv(Path_file, header=None)
        # key_index = {df.iloc[0,i]: i for i in range(df.shape[1])}
        key_index = {df.iat[0, i]: i for i in range(df.shape[1])}
        return key_index, df.iloc[0, :], (df.shape[0]-1), df.shape[1],

    # 不存在列名
    else:
        df = pd.read_csv(Path_file, header=None)
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
    :param key_index: 存储列名与下标字典 key_index
    :param isColumnName:  文件是否有列名
    :param args: 若用户未选择，args为空，则默认最后一列为label，其它列为特征。
                 若用户选择只选择某列当标签，应该传入 一个存有标签列名/索引的包含一个元素的list e.g. [2]
                若用户选择某列为标签，某些列为特征，应该传入 一个存有标签列名/索引的list，和一个存有特征列名/索引的列表list
    :return: 元组(特征 np.array(x), 标签 np.array(y)), 特征列数x.shape[1]
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
            assert len(args[0]) < 3, 'args,传入参数，应该小于3'

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
            assert len(args[0]) < 3, 'args,传入参数，应该小于3'


if __name__ =='__main__':

    path = './model_save/model_params.pkl'

    """         *** Test  Class***         """
    """ case1 """
    # dataFram = pd.read_csv('../data/iris1.data',header=None,)
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

    # data_y, pred_y = rnn.Flow(
    #                         data=data, Seq=1, window_size=1, K_fea=k_fea, HIDDEN_SIZE=20, OUTPUT_SIZE=2, PATH=path,
    #                         num_epochs=20, LR=0.01,isClassfier=True, MODEL='LSTM', BATCH_SIZE_TRA=4, BATCH_SIZE_VAL=1,
    #                         BATCH_SIZE_TES=1)

    """       ************* Test  Regression *************        """
    """ case1 """
    # data_csv = pd.read_csv('../data/data.csv', usecols=[1])
    # dataFram = pd.read_csv('../data/iris1.data',header=None)
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
    # dataFram = pd.read_csv('../data/iris.data', header=None, )
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

    # data_y, pred_y = rnn.Flow(data=data,Seq=1,window_size=2, K_fea=k_fea,HIDDEN_SIZE=20,OUTPUT_SIZE=1,PATH=path,num_epochs=20,LR=0.01,
    #                       isClassfier=False,MODEL='LSTM',BATCH_SIZE_TRA=4,BATCH_SIZE_VAL=1,BATCH_SIZE_TES=1)


    """ Test data_read"""
    # res = data_read_csv(True,'../data/iris1.data')
    # receive_read_data(True,res)

    res = data_read_csv(False, '../data/iris.data')
    receive_read_data(False, res)



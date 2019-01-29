# -*- coding: utf-8 -*-
"""
Created on Fri Jan   2019

@author: Yangkun Cao
"""

import RNN1DBatchNoTrans as RNNnoTRA
import numpy as np
import pandas as pd


class DLRegression():
    '''
     RNN1DBatchNoTrans 时序性分类模型(不同批batch之间不后传)
     回归，输出神经元个数 =1
    '''

    name = 'DLClassification'


    #  add isColumnName(是否有列名)
    def get_config(self, config_info):
        '''
        获取参数信息，config_info中包括model_path（字符串类型）和model_param（字典类型）
        :param config_info: 包含了数据的配置信息，主要是model_path(模型路径)，
                            features_col(特征列名)，label_col(标签列名)，isColumnName(是否有列名)，剩余为model_param
        :return: 返回model_path和model_param
        '''
        if isinstance(config_info, dict):
            is_train = config_info.get('is_train', True)
            # print(is_train)
            if is_train in ['True', '1']:
                is_train = True
            elif is_train in ['False', '0']:
                is_train = False
            config_info.pop('is_train')
            # print(is_train)

            func_name = config_info['func_name']
            config_info.pop('func_name')

            if 'model_path' in config_info.keys():
                model_path = config_info['model_path']
                config_info.pop('model_path')
            else:
                model_path = "classification_model"

            isColumnName = config_info.get('isColumnName', True)
            if isColumnName in ['True', '1']:
                isColumnName = True
            elif isColumnName in ['False', '0']:
                isColumnName = False
            config_info.pop('isColumnName')

            if 'label_col' in config_info.keys():
                label_col = config_info['label_col']
                config_info.pop('label_col')
                if type(label_col) == list and len(label_col) > 0:
                    label_col = [label_col[0]]
                else:
                    label_col = None
            else:
                label_col = None

            # 读取features_col必须在label_col之后执行，顺序不能颠倒
            if 'features_col' in config_info.keys():
                features_col = config_info['features_col']
                config_info.pop('features_col')
                # 可能得到的是一个空的list，所以需要判断一下
                if len(features_col) == 0:
                    features_col = None
            else:
                features_col = None
            # if label_col in features_col:
            #     features_col.remove(label_col)
            # 把剩余的作为模型参数输出
            model_param = config_info
            # print(features_col)
            # print(label_col)
        return is_train, func_name, features_col, label_col, model_path, model_param, isColumnName


    def data_processing(self, dataFrame, isColumnName, features_col, label_col, is_train=True):
        """
        目的，根据用户选择列名或下标，选取特征和标签

       :param dataFrame: 传入之前pandas已读取文件的DataFram，原因在于文件较大时，重新读入文件，浪费时间。
                           | 也可重新读 dataFrame = pd.readcsv(...)，需重新构建key_index。
        :param isColumnName:  文件是否有列名
        :param features_col: 特征列
        :param label_col: 标签列
        :param is_train: 是否是训练

        isColumnName 为True， features_col / label_col 传入得应是字符
        isColumnName 为False， features_col / label_col 传入得应是列表
        若用户未选择，label_col/label_col都为空，则默认最后一列为label，其它列为特征
        若用户选择只选择某列当标签(默认余下列为特征)，应该传入 标签列名(字符)/索引(列表)
        若用户选择某列为标签，某些列为特征，应该传入 存有标签列名(字符)/索引(列表)，和一个存有特征列名(字符)/索引(列表)
        若用户只选择了某些列为特征，未选择特征。 is_train=True 报错：训练时，features_col 存在，label_col 标签列,不能为空
                                                is_train=False 返回特征x ,空列表[]， 特征列数 x.shape[1] (测试时，数据应无标签)

        :return: 数据元组(特征 np.array(x), 标签 np.array(y)), 特征列数x.shape[1]
                  若输入有误，则报错。
        """
        # 存在列名，用户返回应是列名，再寻找索引
        if isColumnName:
            if features_col is None and label_col is None:
                x = dataFrame.iloc[1:, :-1]
                y = dataFrame.iloc[1:, -1]
                return (np.array(x), np.array(y)), x.shape[1]
            elif features_col is None and label_col is not None:
                key_index = {dataFrame.iat[0, i]: i for i in range(dataFrame.shape[1])}
                # key_index = kwargs['key_index']
                # print(args[0])

                index_clo = key_index[label_col]
                y = dataFrame.iloc[1:, index_clo]
                x = pd.concat((dataFrame.iloc[1:, :index_clo], dataFrame.iloc[1:, (index_clo + 1):]), axis=1)
                return (np.array(x), np.array(y)), x.shape[1]
            elif features_col is not None and label_col is not None:
                key_index = {dataFrame.iat[0, i]: i for i in range(dataFrame.shape[1])}
                # key_index = kwargs['key_index']
                label_col = label_col[0]
                index_y = key_index[label_col]
                index_x = [key_index[i] for i in features_col]
                y = dataFrame.iloc[1:, index_y]
                x = pd.concat((dataFrame.iloc[1:, i] for i in index_x), axis=1)
                return (np.array(x), np.array(y)), x.shape[1]
            # elif is_train and features_col is not None and label_col is  None:
            elif is_train:
                assert label_col is not None,'训练时，features_col 存在，label_col 标签列,不能为空'
            else:
                key_index = {dataFrame.iat[0, i]: i for i in range(dataFrame.shape[1])}
                index_x = [key_index[i] for i in features_col]
                x = pd.concat((dataFrame.iloc[1:, i] for i in index_x), axis=1)
                return (np.array(x), np.array([])), x.shape[1]

        # 不存在列名，用户返回应是索引
        else:
            if features_col is None and label_col is None:
                x = dataFrame.iloc[:, :-1]
                y = dataFrame.iloc[:, -1]
                return (np.array(x), np.array(y)), x.shape[1]
            elif features_col is None and label_col is not None:
                index_clo = label_col[0]
                y = dataFrame.iloc[:, index_clo]
                x = pd.concat((dataFrame.iloc[:, :index_clo], dataFrame.iloc[:, (index_clo + 1):]), axis=1)
                return (np.array(x), np.array(y)), x.shape[1]
            elif features_col is not None and label_col is not None:
                index_y = label_col[0]
                index_x = features_col
                y = dataFrame.iloc[:, index_y]
                x = pd.concat((dataFrame.iloc[:, i] for i in index_x), axis=1)
                return (np.array(x), np.array(y)), x.shape[1]
            elif is_train:
                assert label_col is not None,'features_col 存在，label_col 标签列,不能为空'
            else:
                index_x = features_col
                x = pd.concat((dataFrame.iloc[:, i] for i in index_x), axis=1)
                return (np.array(x), np.array([])), x.shape[1]



    def common_parm(self,model_param):
        if 'window_size' in model_param.keys():
            window_size = int(model_param['window_size'])
            # model_param.pop('window_size')
        else:
            window_size = 1

        if 'USE_CUDA' in model_param.keys():
            USE_CUDA = model_param['USE_CUDA']
            if USE_CUDA in ['True', '1']:
                USE_CUDA = True
            elif USE_CUDA in ['False', '0']:
                USE_CUDA = False
            # model_param.pop('USE_CUDA')
        else:
            USE_CUDA = False

        if 'isBatchTes' in model_param.keys():
            isBatchTes = model_param['isBatchTes']
            if isBatchTes in ['True', '1']:
                isBatchTes = True
            elif isBatchTes in ['False', '0']:
                isBatchTes = False
            # model_param.pop('isBatchTes')
        else:
            isBatchTes = False

        # 此文件是回归，可以不传 这个key，默认False
        if 'isClassfier' in model_param.keys():
            isClassfier = model_param['isClassfier']
            if isClassfier in ['True', '1']:
                isClassfier = True
            elif isClassfier in ['False', '0']:
                isClassfier = False
            # model_param.pop('isClassfier')
        else:
            isClassfier = False

        return USE_CUDA, isBatchTes, isClassfier, window_size


    def train_parm(self,model_param):
        if 'HIDDEN_SIZE' in model_param.keys():
            HIDDEN_SIZE = int(model_param['HIDDEN_SIZE'])
            # model_param.pop('HIDDEN_SIZE')
        else:
            HIDDEN_SIZE = 20

        # OUTPUT_SIZE should = 1
        if 'OUTPUT_SIZE' in model_param.keys():
            OUTPUT_SIZE = int(model_param['OUTPUT_SIZE'])
            # model_param.pop('OUTPUT_SIZE')
        else:
            OUTPUT_SIZE = 1

        if 'num_epochs' in model_param.keys():
            num_epochs = int(model_param['num_epochs'])
            # model_param.pop('num_epochs')
        else:
            num_epochs = 10

        if 'LR' in model_param.keys():
            LR = float(model_param['LR'])
            # model_param.pop('LR')
        else:
            LR = 0.1

        if 'func_name' in model_param.keys():
            func_name = model_param['func_name']
            # model_param.pop('func_name')
        else:
            func_name = 'RNN'

        if 'BATCH_SIZE_TRA' in model_param.keys():
            BATCH_SIZE_TRA = int(model_param['BATCH_SIZE_TRA'])
            # model_param.pop('BATCH_SIZE_TRA')
        else:
            BATCH_SIZE_TRA = 4

        if 'BATCH_SIZE_VAL' in model_param.keys():
            BATCH_SIZE_VAL = int(model_param['BATCH_SIZE_VAL'])
            # model_param.pop('BATCH_SIZE_VAL')
        else:
            BATCH_SIZE_VAL = 1

        if 'BATCH_SIZE_TES' in model_param.keys():
            BATCH_SIZE_TES = int(model_param['BATCH_SIZE_TES'])
            # model_param.pop('BATCH_SIZE_TES')
        else:
            BATCH_SIZE_TES = 1

        if 'seq' in model_param.keys():
            seq = int(model_param['seq'])
            # model_param.pop('seq')
        else:
            seq = 1

        if 'CUDA_ID' in model_param.keys():
            CUDA_ID = model_param['CUDA_ID']
            # model_param.pop('CUDA_ID')
        else:
            CUDA_ID = "0"

        if 'BATCH_SIZE_TES' in model_param.keys():
            BATCH_SIZE_TES = int(model_param['BATCH_SIZE_TES'])
            # model_param.pop('seq')
        else:
            BATCH_SIZE_TES = 1

        return  HIDDEN_SIZE, OUTPUT_SIZE, num_epochs, LR,func_name, BATCH_SIZE_TRA, \
               BATCH_SIZE_VAL, BATCH_SIZE_TES,  CUDA_ID, BATCH_SIZE_TES


    def test_parm(self,model_param):
        if 'CUDA_ID' in model_param.keys():
            CUDA_ID = model_param['CUDA_ID']
            # model_param.pop('CUDA_ID')
        else:
            CUDA_ID = "0"

        if 'BATCH_SIZE_TES' in model_param.keys():
            BATCH_SIZE_TES = int(model_param['BATCH_SIZE_TES'])
            # model_param.pop('seq')
        else:
            BATCH_SIZE_TES = 1
        return CUDA_ID, BATCH_SIZE_TES

    # def run(self,app, data_key=None, data_path=None, config_info=None, only_return_df=False):
    def run(self, data_path=None, config_info=None):
        '''
        执行分类算法，可以执行多种分类算法
        :param data_key: 输入数据在内存中的地址，输入类型为字符串
        :param data_path: 数据的保存路径，输入类型为字符串
        :param config_info: 配置信息，其中包括is_train，func_name，model_path，label_col，
                            features_col，model_param，isColumnName
        :param only_return_df: 判断是否返回预览信息
        :return:
        '''
        # 读取数据
        # 需考虑数据库，内存地址
        dataFrame = pd.read_csv(data_path, header=None)


        # 读取配置
        is_train, func_name, features_col, label_col, model_path, model_param, isColumnName = \
            self.get_config( config_info=config_info)
        # 数据处理

        data, k_fea = self.data_processing(dataFrame, isColumnName, features_col, label_col, is_train)

        # 修改参数中的数值为float类型，因为传入的时候都是string
        # model_param = self.modify_param(model_param)
        # print(model_param)

        USE_CUDA, isBatchTes, isClassfier, window_size = self.common_parm(model_param)

        if is_train:
            try:
                HIDDEN_SIZE, OUTPUT_SIZE, num_epochs, LR, func_name, \
                BATCH_SIZE_TRA, BATCH_SIZE_VAL, BATCH_SIZE_TES,CUDA_ID, BATCH_SIZE_TES = self.train_parm(model_param)

                data_y, pred_y = RNNnoTRA.Flow_load(data=data, window_size=window_size, K_fea=k_fea, HIDDEN_SIZE=HIDDEN_SIZE,
                              OUTPUT_SIZE=OUTPUT_SIZE, PATH=model_path,num_epochs=num_epochs, LR=LR,isClassfier=isClassfier,
                              MODEL=func_name, BATCH_SIZE_TRA=BATCH_SIZE_TRA, BATCH_SIZE_VAL=BATCH_SIZE_VAL,
                            BATCH_SIZE_TES=BATCH_SIZE_TES,USE_CUDA=USE_CUDA,isBatchTes=isBatchTes,CUDA_ID=CUDA_ID)

            except AttributeError:
                print("No model named " + func_name)
        else:
            try:
                CUDA_ID, BATCH_SIZE_TES = self.test_parm(model_param)
                data = data[0]  # get feature
                # print(data)
                pred_y = RNNnoTRA.load_model_test_data(model_path,data,isClassfier=isClassfier,isBatchTes=isBatchTes,
                                              Seq=window_size,K_fea=k_fea,CUDA_ID=CUDA_ID,BATCH_SIZE_TES=BATCH_SIZE_TES,USE_CUDA=USE_CUDA)
                print(pred_y)
            except AttributeError:
                print("No model named " + func_name)


        # 模型评估 应加
if __name__ == '__main__':
    path_column_T = 'iris1.data'
    # test data_processing
    # isColumnName = True 则features_col， label_col 里应该是字符
    # 'model_path': 'model_params_regression.pkl'
    config_info1T = {'BATCH_SIZE_TES': '1','USE_CUDA': '0','isBatchTes': 'False','window_size': '1','HIDDEN_SIZE': '20',
                   'model_path': 'model_params_regression.pkl','isColumnName':'True',
                   'func_name': 'RNN', 'is_train': 'True',
                   'features_col': [], 'tol': '1e-06', 'label_col': [],}
    config_info2T = {'BATCH_SIZE_TES': '1', 'USE_CUDA': '0', 'isBatchTes': 'False', 'window_size': '1',
                    'HIDDEN_SIZE': '20',
                    'model_path': 'model_params_regression.pkl', 'isColumnName': 'True',
                    'func_name': 'RNN', 'is_train': 'True',
                    'features_col': ['fea1'], 'tol': '1e-06', 'label_col': [], }
    config_info3T = {'BATCH_SIZE_TES': '1', 'USE_CUDA': '0', 'isBatchTes': 'False', 'window_size': '1',
                    'HIDDEN_SIZE': '20',
                    'model_path': 'model_params_regression.pkl', 'isColumnName': 'True',
                    'func_name': 'RNN', 'is_train': 'True',
                    'features_col': ['fea1'], 'tol': '1e-06', 'label_col': ['lab'], }
    config_info4T = {'BATCH_SIZE_TES': '1', 'USE_CUDA': '0', 'isBatchTes': 'False', 'window_size': '1',
                     'HIDDEN_SIZE': '20',
                     'model_path': 'model_params_regression.pkl', 'isColumnName': 'True',
                     'func_name': 'RNN', 'is_train': 'True',
                     'features_col': ['fea1','fea2'], 'tol': '1e-06', 'label_col': ['lab'], }
    # isColumnName = False 则features_col， label_col 里应该是索引
    path_column_F = 'iris.data'
    config_info1F = {'BATCH_SIZE_TES': '1', 'USE_CUDA': '0', 'isBatchTes': 'False', 'window_size': '1',
                    'HIDDEN_SIZE': '20',
                    'model_path': 'model_params_regression.pkl', 'isColumnName': 'False',
                    'func_name': 'RNN', 'is_train': 'True',
                    'features_col': [], 'tol': '1e-06', 'label_col': [4], }
    config_info2F = {'BATCH_SIZE_TES': '1', 'USE_CUDA': '0', 'isBatchTes': 'False', 'window_size': '1',
                     'HIDDEN_SIZE': '20',
                     'model_path': 'model_params_regression.pkl', 'isColumnName': 'False',
                     'func_name': 'RNN', 'is_train': 'True',
                     'features_col': [], 'tol': '1e-06', 'label_col': [], }
    config_info3F = {'BATCH_SIZE_TES': '1', 'USE_CUDA': '0', 'isBatchTes': 'False', 'window_size': '1',
                     'HIDDEN_SIZE': '20',
                     'model_path': 'model_params_regression.pkl', 'isColumnName': 'False',
                     'func_name': 'RNN', 'is_train': 'True',
                     'features_col': [1], 'tol': '1e-06', 'label_col': [4], }
    config_info4F = {'BATCH_SIZE_TES': '1', 'USE_CUDA': '0', 'isBatchTes': 'False', 'window_size': '1',
                     'HIDDEN_SIZE': '20',
                     'model_path': 'model_params_regression.pkl', 'isColumnName': 'False',
                     'func_name': 'RNN', 'is_train': 'True',
                     'features_col': [1,2], 'tol': '1e-06', 'label_col': [4], }

    """case1"""
    # DLRegression().run(path_column_T,config_info=config_info2T)
    """case2"""
    # DLRegression().run(path_column_F, config_info=config_info4F)

    """case3 is_train = False"""
    config_info4F_3 = {'BATCH_SIZE_TES': '1', 'USE_CUDA': '0', 'isBatchTes': 'False', 'window_size': '1',
                     'HIDDEN_SIZE': '20',
                     'model_path': 'model_params_regression.pkl', 'isColumnName': 'False',
                     'func_name': 'RNN', 'is_train': 'False',
                     'features_col': [1, 2], 'tol': '1e-06', 'label_col': [4], }
    # DLRegression().run(path_column_F, config_info=config_info4F_3)
    config_info4T_3 = {'BATCH_SIZE_TES': '1', 'USE_CUDA': '0', 'isBatchTes': 'False', 'window_size': '1',
                     'HIDDEN_SIZE': '20','num_epochs':'100','LR':'0.1',
                     'model_path': 'model_params_regression.pkl', 'isColumnName': 'True','isClassfier':'False',
                     'func_name': 'RNN', 'is_train': 'False',
                     'features_col': ['fea1', 'fea2'], 'tol': '1e-06', 'label_col': ['lab'], }
    DLRegression().run(path_column_T, config_info=config_info4T_3)

    # from get_offline_app import GetSparkContextSession
    # data_path = 'hdfs://phm1:8020/user/root/test/kang_clasification.csv'
    # data_key = "kang"
    # mlclassification_config = {
    #     "features_col": ["drive_end_data", "fan_end_data"],
    #     "is_train": "True",
    #     "label_col": "fault_status",
    #     "model_path": "hdfs://10.28.0.191:8020/user/root/test/compute_feature/regressor_models",
    #     "func_name": "OneVsRest"
    # }
    #
    # res = MLClassification.run(app, data_key=data_key, data_path=data_path, config_info=mlclassification_config, only_return_df=False)
    # # print(res)
    # app = get_app(sc, spark)
    # app = app
    # data_path = "hdfs://10.28.0.191:8020/user/root/test/public.cwru.parquet"
    # data_key = "data_key#1_data_key#1_public.cwru.csv"
    # config_info = {'regParam': '0.0',
    #                'model_path': 'hdfs://10.28.0.191:8020/user/root/test/算法模型#LogisticRegression_用户hd_工程42_1545276395',
    #                'func_name': 'LogisticRegression', 'is_train': 'True',
    #                'features_col': ['fan_end_data', 'rotate_speed'], 'tol': '1e-06', 'label_col': ['drive_end_data'],
    #                'maxIter': '100', 'threshold': '0.5'}
    # res = MLClassification().run(app, data_key=data_key, data_path=data_path, config_info=config_info)


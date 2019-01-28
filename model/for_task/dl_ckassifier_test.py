# -*- coding: utf-8 -*-
"""
Created on Fri Jan   2019

@author: Ran
"""
# from public_methods import PublicMethods
# from pyspark.ml.feature import VectorAssembler
# import pyspark.ml.classification
# from evaluate_model import EvaluateModel
# import json
import RNN1DBatchNoTrans as RNNnoTRA
import numpy as np
import pandas as pd


class DLClassification():
    '''
     RNN1DBatchNoTrans 时序性分类模型(不同批batch之间不互传)
    '''

    name = 'DLClassification'

    def get_config(self, data, config_info):
        '''
        获取参数信息，config_info中包括model_path（字符串类型）和model_param（字典类型）
        :param config_info: 包含了数据的配置信息，主要是model_path(模型路径)，
                            features_col(特征列名)，label_col(标签列名)，剩余为model_param
        :return: 返回model_path和model_param
        '''
        if isinstance(config_info, dict):
            is_train = config_info.get('is_train', True)
            if is_train in ['True', '1']:
                is_train = True
            elif is_train in ['False', '0']:
                is_train = False
            config_info.pop('is_train')

            func_name = config_info['func_name']
            config_info.pop('func_name')

            if 'model_path' in config_info.keys():
                model_path = config_info['model_path']
                config_info.pop('model_path')
            else:
                model_path = "classification_model"

            if 'label_col' in config_info.keys():
                label_col = config_info['label_col']
                config_info.pop('label_col')
                if type(label_col) == list:
                    label_col = label_col[0]
            else:
                label_col = None

            # 读取features_col必须在label_col之后执行，顺序不能颠倒
            if 'features_col' in config_info.keys():
                features_col = config_info['features_col']
                config_info.pop('features_col')
                # 可能得到的是一个空的list，所以需要判断一下
                if len(features_col) == 0:
                    features_col = data.columns
            else:
                features_col = data.columns
            if label_col in features_col:
                features_col.remove(label_col)
            # 把剩余的作为模型参数输出
            model_param = config_info
        return is_train, func_name, features_col, label_col, model_path, model_param


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

    def get_load_model_func_name(self, func_name):
        '''
        对模型测试的时候，通过给出的func_name找到对应的加载模型的函数名
        :param func_name: 函数名，输入类型为字符串
        :return:
        '''
        algorithm_name = {'RNN': 'RNN',
                          'LSTM': 'LSTM',}
        func_name = algorithm_name[func_name]
        return func_name

    def modify_param(self, model_param):
        '''
        将参数中的数值都改变成float类型，因为从数据库中传入的时候，参数默认是string类型
        :param model_param: 模型参数，字典类型
        :return: 将数值型的数据改成float类型返回
        '''
        # 如果使用默认参数，那么model_param是一个空值，所以先判断model_param的长度
        if isinstance(model_param, dict) and len(model_param) > 0:
            for key in model_param.keys():
                temp = model_param[key]
                try:
                    float(temp)
                    print(float(temp))
                except:
                    pass
                else:
                    model_param[key] = float(temp)
        return model_param

    def modify_model_param(self, is_train, model_param, label_col, df_spark, features_col):
        '''
        为数据的训练和测试配一些参数
        :param is_train: 用来判断是否是训练还是测试，输入类型为bool类型True或者False
        :param model_param: 模型参数列表，输入的类型为字典dict
        :param label_col: 标签列，输入的类型为字符串
        :param df_spark: 输入数据，类型为spark的DataFrame
        :param features_col: 特征列，输入类型为字符串列表
        :return:
        '''
        # 如果是训练集为模型设置label和prediction的预测名，prediction训练是什么样子测试就是什么样子
        if is_train is True:
            # 为模型参数添加列名和特征名字
            model_param['labelCol'] = label_col
            prediction_col = label_col + '#prediction'
            model_param['predictionCol'] = prediction_col
            # 如果已经预测过一次，将原来的预测删掉，这一步的操作相当于覆盖掉原来的数据，这个主要针对从data_key中读取的数据
            if prediction_col in df_spark.columns:
                df_spark = df_spark.drop(prediction_col)
            if prediction_col in features_col:
                features_col.remove(prediction_col)
            print('去除df_spark后的列名:', df_spark.columns)
            print('预测列名prediction_col:', prediction_col)

        features_vector = ''
        for i in features_col:
            features_vector = features_vector + i
        features_vector = features_vector + '#Vector'
        print(features_vector)
        model_param['featuresCol'] = features_vector
        # 转换数据类型
        df_spark = PublicMethods().data_type_transform(df_spark, features_col, to_type='double')
        #df_spark.show()
        # 生成特征向量
        assembler = VectorAssembler().setInputCols(features_col).setOutputCol(features_vector)
        assembler.getOutputCol()
        df_spark = assembler.transform(df_spark)
        # 读取列名
        all_columns = df_spark.columns
        print('df_spark.all_columns', all_columns)
        prediction_col = [prediction_col]#列名最好为list
        return df_spark, model_param, all_columns,prediction_col

    def run(self,app, data_key=None, data_path=None, config_info=None, only_return_df=False):
        '''
        执行分类算法，可以执行多种分类算法
        :param data_key: 输入数据在内存中的地址，输入类型为字符串
        :param data_path: 数据的保存路径，输入类型为字符串
        :param config_info: 配置信息，其中包括is_train，func_name，model_path，label_col，
                            features_col，model_param
        :param only_return_df: 判断是否返回预览信息
        :return:
        '''
        # 读取数据
        # app spark-shell全局变量
        df_spark = PublicMethods().get_data(app=app, data_path=data_path, data_key=data_key)
        original_columns = df_spark.columns
        df_spark = PublicMethods().data_type_transform(df_spark, original_columns, to_type='float')

        # 读取配置
        is_train, func_name, features_col, label_col, model_path, model_param = \
            self.get_config(data=df_spark, config_info=config_info)
        df_spark = PublicMethods().data_type_transform(df_spark, label_col, to_type='int')

        # 修改参数中的数值为float类型，因为传入的时候都是string
        model_param = self.modify_param(model_param)
        print(model_param)
        # 如果是训练集为模型设置label和prediction的预测名，prediction训练是什么样子测试就是什么样子
        df_spark, model_param, all_columns,prediction_col = \
            self.modify_model_param(is_train, model_param, label_col, df_spark, features_col)

        if is_train:
            try:
                func_obj = getattr(pyspark.ml.classification, func_name)
                model = func_obj(**model_param)
                model = model.fit(df_spark)
                model.write().overwrite().save(model_path)
                # print('------------------------------', func_obj)
            except AttributeError:
                print("No model named " + func_name)
        else:
            try:
                func_name = self.get_load_model_func_name(func_name)
                func_obj_model = getattr(pyspark.ml.classification, func_name)
                model = func_obj_model.load(model_path)
            except AttributeError:
                print("No model named " + func_name)

        df_re = model.transform(df_spark)
        print('-------------------------------', '输出预测后的数据！！！！！')
        #df_re.show()
        # # 得到预测后的列名
        # prediction_col = list(set(df_re.columns).difference(set(all_columns)))
        # # 把多余的删除
        # for col in prediction_col:
        #     if '#' not in col:
        #         prediction_col.remove(col)
        # print('---------------------------------', prediction_col)
        # 返回原始数据和预测得到的列
        print('original_columns:',original_columns)
        print('prediction_col:',prediction_col)
        print('original_columns+prediction_col:',original_columns+prediction_col)
        tmp_list  = list(set(original_columns+prediction_col))
        df_re = df_re.select(tmp_list)
        print('columns:', df_re.columns)
        # 全部数据写入data_key
        PublicMethods().cache_data(app=app, data=df_re, data_key=data_key, operate="write")
        #df_re.show()

        if only_return_df:
            return df_re
        else:
            evaluation = {}
            # 只有历史流程需要model evaluation，实时流程不需要
            eval_config = {
                "prediction_col": PublicMethods().transform_to_list(prediction_col),
                "label_col": PublicMethods().transform_to_list(label_col),
                "metric_name": ["f1", "weightedPrecision", "weightedRecall", "areaUnderROC", "areaUnderPR"]
            }
            print('prediction_col',prediction_col)
            print('label_col',label_col)
            print('eval_config',eval_config)
            res1 = EvaluateModel().run(data=df_re, config_info=eval_config, use_app=True)
            res1 = json.loads(res1)
            evaluation = res1['evaluation']
            pred_list = df_re.select(prediction_col).toPandas().values.tolist()
            label_list = df_re.select(PublicMethods().transform_to_list(label_col)).toPandas().values.tolist()
            print("label_list:", label_list)
            print("pred_list:", pred_list)
            if 'confusion_matrix' not in evaluation.keys():
                evaluation['confusion_matrix'] = PublicMethods().get_confusion_matrix(df_re, prediction_col, label_col)

            data_preview = df_re.select(prediction_col)
            data_preview = data_preview.toPandas().head(100).values.tolist()
            data_preview.insert(0, prediction_col)
            return_dict = PublicMethods().get_return_dict(data_preview={"data": data_preview}, data_key=data_key,
                                                          model_path=model_path, output_columns=prediction_col,
                                                          func_return_args=data_preview,evaluation=evaluation)
            print(return_dict)
            return return_dict
if __name__ == '__main__':
    from get_offline_app import GetSparkContextSession
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
    app = app
    data_path = "hdfs://10.28.0.191:8020/user/root/test/public.cwru.parquet"
    data_key = "data_key#1_data_key#1_public.cwru.csv"
    config_info = {'regParam': '0.0',
                   'model_path': 'hdfs://10.28.0.191:8020/user/root/test/算法模型#LogisticRegression_用户hd_工程42_1545276395',
                   'func_name': 'LogisticRegression', 'is_train': 'True',
                   'features_col': ['fan_end_data', 'rotate_speed'], 'tol': '1e-06', 'label_col': ['drive_end_data'],
                   'maxIter': '100', 'threshold': '0.5'}
    res = MLClassification().run(app, data_key=data_key, data_path=data_path, config_info=config_info)


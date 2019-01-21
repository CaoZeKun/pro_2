

# 时间序列模型


@[TOC](目录)
## Requirements
时间序列模型需要以下才可以运行：
 1. python 3.0+
 2. pytorch 1.0.0 (pytorch安装说明见pytorch_install_step文件)
 3. numpy (1.14.6 服务器环境 - 实际要求可能不需要这么高)
 4. pandas (0.23.4 服务器环境 - 实际要求可能不需要这么高)
## Usage
由于不同情况导致用法与问题不同，写了两个模块，详情如下：

### 一 RNN_1D_batch_no_trans.py
前后Batch 之间的 hidden_state未相互传递
	主要函数
 1. data_read_csv 目的：读取文件
 2. data_return 目的：返回给前台列名(有多种返回值)，以便选取特征和标签 (未考虑特征列名有重复)
 3. data_processing 目的，根据用户选择列名或下标，选取特征和标签
 		需注意以下传入参数详细说明：
 		:param args: 若用户未选择，args为空，则默认最后一列为label，其它列为特征。若用户选择只选择某列当标签，应该传入 一个存有标签列名/索引的包含一个元素的list。 E.G.若用户选择某列为标签，某些列为特征，应该传入 一个存有标签列名/索引的list，和一个存有特征列名/索引的列表list
    :param kwargs: 应当传入的是存储列名与下标字典 key_index， 得到的是{key_index ： key_index}

4. create_dataset 目的：处理数据，使用连续的window_size个时间步样本作为特征，最后一个时间步样本的真实值作为标签。
5. load_data_loader 目的：装载训练/验证/测试数据
6. construct_model_opt 目的：构建模型，选择优化算法，选择损失函数。
7. train_model 目的：训练模型，分类保存准确率最高模型，回归则保存loss最低。
8. load_model_test 目的：加载训练好的模型，进行测试/预测
9. Flow 目的：整体流程: 数据装载 -> 模型构建 -> 模型训练(保存)->模型测试。 (Flow函数没有把所有参数全部传入，若有必要改写为**kwargs)

### 二 RNN_1D_batch_trans.py
前后Batch 之间的 hidden_state相互传递，但Batch最后一批若不够Batch_Size应舍弃，且建议Batch不打乱。(未列出的函数与 一 RNN_1D_batch_no_trans相同)
**RNN/LSTM 模型传入数据不同，需多传一个hidden_state参数**
	主要不同函数
5. load_data_loader 目的：装载训练/验证/测试数据 (是否丢弃最后一组不够一个批量的样本，默认True)
7. train_model 目的：训练模型，分类保存准确率最高模型，回归则保存loss最低。(由于RNN/LSTM传参不同，会多判断一次模型名称，且RNN/LSTM模型传参与另一版本不同）
8. load_model_test 目的：加载训练好的模型，进行测试/预测(更改主要在于RNN/LSTM模型传参)
9. Flow 目的：整体流程: 数据装载 -> 模型构建 -> 模型训练(保存)->模型测试。 
### 三 测试样例
 if name == __main__包含(data_processing数据处理) 16种情况的测试，
 * 分类(代码通过标志位判断是否数据有列名，只是为了测试，分为两个Test)
 	* Test1 传回参数 为用户列名 (数据文件有列名)
 		* 若用户未选择，args为空，则默认最后一列为label，其它列为特征。
 		*  若用户选择只选择某列当标签(默认余下列为特征)，应该传入 一个存有标签**列名**的包含一个元素的list。  e.g.  \[2]
 		* 若用户选择某列为标签，某些列为特征，应该传入 一个存有标签**列名**的list，和一个存有特征**列名**的列表list
 		* 若参数传入错误
 	* Test2 传回参数 为用户选择下标 (数据文件无列名)
 		* 若用户未选择，args为空，则默认最后一列为label，其它列为特征。
 		*  若用户选择只选择某列当标签(默认余下列为特征)，应该传入 一个存有标签**索引**的包含一个元素的list。  e.g.  \[2]
 		* 若用户选择某列为标签，某些列为特征，应该传入 一个存有标签**索引**的list，和一个存有特征**索引**的列表list
 		* 若参数传入错误
* 回归：(需注意回归输出神经元应当为1)
	* Test1 -与分类类似 
	* Test2 -与分类类似 
 
some situation attention
* 虽然使用 window_size增加 seq维度特征，但测试时候seq维度应为1.
* 回归的标签应为 FloatTensor，分类标签应为LongTensor




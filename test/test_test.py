import numpy as np
data = np.random.rand(10,5)

data_x = data[:,:-1, np.newaxis]
print(data_x.ndim)

# x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
# y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

class MyError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

try:
    raise MyError(2*2)
except MyError as e:
    print('My exception occurred, value:', e.value)
"""
# _*_coding=UTF-8_*_
# #使用自定义异常类实现指定输入字符串长度
# #自定义异常类
class SomeCustomError(Exception):
    def __init__(self,str_length):
        super(SomeCustomError,self).__init__()
        self.str_length = str_length
#使用自定义异常
# length = input("输入指定输入字符串长度范围:\n")
length = 3
while True:
    try:
        # s = raw_input("输入一行字符串:\n")
        s = 'sadsadasfsafsafsa'
        #输入字符串长度超过指定长度范围,引发异常
        if (length < len(s)):
                raise SomeCustomError(length)
    except SomeCustomError as x:
        print("捕获自定义异常")
        print("输入字符串重读应该小于%d,请重新输入!") % x.str_length
    else:
        print("输入字符串为%s") % s
        break
"""

length=2
try:
    # s = raw_input("输入一行字符串:\n")
    s = 'sadsadasfsafsafsa'
    # 输入字符串长度超过指定长度范围,引发异常
    if (length < len(s)):
        raise length
except :
    print("捕获自定义异常")


# python 3.6
# 定义一个新的函数
def printStar(func):
    print('*************************************')
    return func()


@printStar
def add():
    return 1 + 1


def sub():
    return 2 - 1


print(add)
print(printStar(sub))




# python 3.6
# 定义一个新的函数
def printStar(func):
    def f():
        print('*************************************')
        return func()

    return f


@printStar
def add():
    return 1 + 1


@printStar
def sub():
    return 2 - 1


print(add())

print(sub())






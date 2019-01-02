import numpy as np
import torch
import matplotlib.pyplot as plt


a = [0,5,10]
b = [1,6,12]

plt.plot(a,b)
plt.show()

I = np.loadtxt('../data/test.txt')
print(I)

aa = ['abc', 'dsd', 'sss']
aa = np.array(aa)
print(aa)
# np.savetxt('../data/test.txt',aa,fmt='%s')


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# I = mpimg.imread('../data/hymenoptera_data/train/ants/0013035.png')
# print(I.shape)
# plt.imshow(I)

# 保存训练的模型 # 保存整个网络和参数
torch.save(net, 'net.pkl')
# 重新加载模型
net = torch.load('net.pkl')
# 用新加载的模型进行预测
prediction = net(x)
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw = 5)
plt.show()


# 只保存网络的参数, 官方推荐的方式
torch.save(net.state_dict(), 'net_params.pkl')
# 定义网络
net = torch.nn.Sequential( torch.nn.Linear(1, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1) )
加载网络参数
net.load_state_dict(torch.load('net_params.pkl'))
# 用新加载的参数进行预测
prediction = net(x) plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw = 5)
plt.show()

"""   """
torch.save(model.state_dict(), PATH)

###
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()

"""   """
torch.save(model, PATH)
# Model class must be defined somewhere
model = torch.load(PATH)
model.eval()





"""   """
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)

###
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()

""""""
torch.save({
            'modelA_state_dict': modelA.state_dict(),
            'modelB_state_dict': modelB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            ...
            }, PATH)

###
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()


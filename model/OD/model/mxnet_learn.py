
from mxnet import image
import gluonbook as gb

gb.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
gb.plt.imshow(img) # 加分号只显示图。


from fastai.vision import *

path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
learn = create_cnn(data, models.resnet18, metrics=accuracy)
learn.fit(2)

# img, label = ds[0]
# img.show(figsize(2,2),title='MNIST digit')
# img.rotate(35)
#
# tfms = [rotate(degrees=(-20,20)), symmetric_warp(magnitude=(-0.3,0.3))]
# fig,axes = plt.subplots(1,4,figsize=(8,2))
# for ax in axes: ds[0][0].apply_tfms(tfms).show(ax=ax)
# data = ImageDataBunch.from_folder(path, ds_tfms=(tfms, []))
# learn = create_cnn(data, models.resnet18, metrics=accuracy)
# learn.fit(1)
# interp = ClassificationInterpretation.from_learner(learn)
# interp.plot_top_losses(9, figsize=(6,6))
# interp.plot_confusion_matrix()
# img = learn.data.train_ds[0][0]
# learn.predict(img)

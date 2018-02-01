import numpy as np
import tensorflow as tf
import os, glob
from skimage import io, transform
# tf.one-hot的使用
CLASS = 6
label1 = tf.constant([0, 4, 5, 4, 4, 0, 1, 1, 1, 2, 5, 4, 5, 3, 0, 1, 0, 4, 3, 5])
sess1 = tf.Session()
print('label1:', sess1.run(label1))
b = tf.one_hot(label1, CLASS, 1, 0)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(b)
    #print('after one_hot', sess.run(b))

path = 'C:/data/test/'
w = 224
h = 224
c = 1


def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):  # cate为6个分类的文件夹路径
        for im in glob.glob(folder + '/*.jpg'):  # 匹配folder路径下的所有,jpg文件
            # print('reading the image: %s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)
            labels.append(idx)  # label以列表形式保存，相当于进行过了one-hot编码
    y = np.asarray(labels, np.int32)
    return np.asarray(imgs, np.float32), y
print(read_img(path))
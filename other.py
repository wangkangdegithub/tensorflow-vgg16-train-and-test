from skimage import io, transform
import glob
import os
import tensorflow as tf
import numpy as np
import time

path = 'C:/data/test/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

# 将所有的图片resize成224*224
w = 224
h = 224
c = 1


# 读取图片
def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))  # 图片resize
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


data, label = read_img(path)

# 打乱顺序
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]

# 将所有数据分为训练集和验证集
ratio = 0.8
s = np.int(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
x_val = data[s:]
y_val = label[s:]

# -----------------构建网络----------------------
# 占位符
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')  # 输入
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')  # 真实的输出


def weight_variable(shape, name="weights"):  # 初始化权重 w
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)  # 不为标准正态分布\截断的正态分布噪声
    return tf.Variable(initial, name=name)


def bias_variable(shape, name="biases"):  # 初始化偏置项 b
    initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(input, w):  # 卷积项，w为卷积核
    return tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='SAME')


def pool_max(input):  # 最大池化项
    return tf.nn.max_pool(input,
                          ksize=[1, 2, 2, 1],  # 池化层尺寸
                          strides=[1, 2, 2, 1],  # 步长尺寸
                          padding='SAME',
                          )


def fc(input, w, b):  # 全连接层
    return tf.matmul(input, w) + b


# conv1
with tf.name_scope('conv1_1') as scope:
    kernel = weight_variable([3, 3, 1, 64])  # 重点关注第一层卷积。卷积核为3*3,输入channels = 1(输入通道数),卷集核个数（输出通道数）为64。
    biases = bias_variable([64])
    output_conv1_1 = tf.nn.relu(conv2d(x, kernel) + biases, name=scope)

with tf.name_scope('conv1_2') as scope:
    kernel = weight_variable([3, 3, 64, 64])
    biases = bias_variable([64])
    output_conv1_2 = tf.nn.relu(conv2d(output_conv1_1, kernel) + biases, name=scope)

pool1 = pool_max(output_conv1_2)

# conv2
with tf.name_scope('conv2_1') as scope:
    kernel = weight_variable([3, 3, 64, 128])
    biases = bias_variable([128])
    output_conv2_1 = tf.nn.relu(conv2d(pool1, kernel) + biases, name=scope)

with tf.name_scope('conv2_2') as scope:
    kernel = weight_variable([3, 3, 128, 128])
    biases = bias_variable([128])
    output_conv2_2 = tf.nn.relu(conv2d(output_conv2_1, kernel) + biases, name=scope)

pool2 = pool_max(output_conv2_2)

# conv3
with tf.name_scope('conv3_1') as scope:
    kernel = weight_variable([3, 3, 128, 256])
    biases = bias_variable([256])
    output_conv3_1 = tf.nn.relu(conv2d(pool2, kernel) + biases, name=scope)

with tf.name_scope('conv3_2') as scope:
    kernel = weight_variable([3, 3, 256, 256])
    biases = bias_variable([256])
    output_conv3_2 = tf.nn.relu(conv2d(output_conv3_1, kernel) + biases, name=scope)

with tf.name_scope('conv3_3') as scope:
    kernel = weight_variable([3, 3, 256, 256])
    biases = bias_variable([256])
    output_conv3_3 = tf.nn.relu(conv2d(output_conv3_2, kernel) + biases, name=scope)

pool3 = pool_max(output_conv3_3)

# conv4
with tf.name_scope('conv4_1') as scope:
    kernel = weight_variable([3, 3, 256, 512])
    biases = bias_variable([512])
    output_conv4_1 = tf.nn.relu(conv2d(pool3, kernel) + biases, name=scope)

with tf.name_scope('conv4_2') as scope:
    kernel = weight_variable([3, 3, 512, 512])
    biases = bias_variable([512])
    output_conv4_2 = tf.nn.relu(conv2d(output_conv4_1, kernel) + biases, name=scope)

with tf.name_scope('conv4_3') as scope:
    kernel = weight_variable([3, 3, 512, 512])
    biases = bias_variable([512])
    output_conv4_3 = tf.nn.relu(conv2d(output_conv4_2, kernel) + biases, name=scope)

pool4 = pool_max(output_conv4_3)

# conv5
with tf.name_scope('conv5_1') as scope:
    kernel = weight_variable([3, 3, 512, 512])
    biases = bias_variable([512])
    output_conv5_1 = tf.nn.relu(conv2d(pool4, kernel) + biases, name=scope)

with tf.name_scope('conv5_2') as scope:
    kernel = weight_variable([3, 3, 512, 512])
    biases = bias_variable([512])
    output_conv5_2 = tf.nn.relu(conv2d(output_conv5_1, kernel) + biases, name=scope)

with tf.name_scope('conv5_3') as scope:
    kernel = weight_variable([3, 3, 512, 512])
    biases = bias_variable([512])
    output_conv5_3 = tf.nn.relu(conv2d(output_conv5_2, kernel) + biases, name=scope)

pool5 = pool_max(output_conv5_3)

# fc6
with tf.name_scope('fc6') as scope:
    shape = int(np.prod(pool5.get_shape()[1:]))
    kernel = weight_variable([shape, 1024])
    biases = bias_variable([1024])
    pool5_flat = tf.reshape(pool5, [-1, shape])
    output_fc6 = tf.nn.relu(fc(pool5_flat, kernel, biases), name=scope)

# drop操作
keep_prob = tf.placeholder(tf.float32)
output_fc6_drop = tf.nn.dropout(output_fc6, keep_prob)

# fc7
with tf.name_scope('fc7') as scope:
    kernel = weight_variable([1024, 1024])
    biases = bias_variable([1024])
    output_fc7 = tf.nn.relu(fc(output_fc6_drop, kernel, biases), name=scope)

# drop操作
output_fc7_drop = tf.nn.dropout(output_fc7, keep_prob)

# fc8
with tf.name_scope('fc8') as scope:
    kernel = weight_variable([1024, 4])  # 重点关注最后一个全连接层，输入为1024个通道数、输出为4个（4分类）
    biases = bias_variable([4])
    output_fc8 = tf.nn.relu(fc(output_fc7_drop, kernel, biases), name=scope)
# ---------------------------网络结束---------------------------

loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=output_fc8)
train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(output_fc8, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):  # start_idx从0开始，每50个一次
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# 训练和测试数据，可将n_epoch设置更大一些

n_epoch = 30
batch_size = 5
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    start_time = time.time()

    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    for i in range(100):  # 每轮epoch，做100轮batch_size，每个batch_size有5个样本的训练
        for x_train_batch, y_train_batch in minibatches(x_train, y_train, batch_size, shuffle=True):
            _, err, ac = sess.run([train_op, loss, acc],
                                  feed_dict={x: x_train_batch, y_: y_train_batch, keep_prob: 1.0})
            train_loss += err
            train_acc += ac
            n_batch += 1

            if n_batch % 5 == 0:  # 在一个batch_size中每训练5个样本后，在测试验证集上观察val_loss和val_acc
                print('第%s轮epoch:' % epoch, "train loss: %f" % (train_loss / n_batch), "train acc: %f" % (train_acc / n_batch))

                # validation
                val_loss, val_acc, n_batch = 0, 0, 0
                for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
                    err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a, keep_prob: 0.5})
                    val_loss += err
                    val_acc += ac
                    n_batch += 1
                print('第%s轮epoch:' % epoch, "val loss: %f" % (val_loss / n_batch), "val acc: %f" % (val_acc / n_batch))

sess.close()

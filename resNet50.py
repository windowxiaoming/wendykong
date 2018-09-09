#-*-coding:utf-8-*-
import numpy as np
import scipy.io as scio
from keras.layers import Conv2D,BatchNormalization,Activation,Add,Flatten,Dense,MaxPooling2D,Input
from keras.models import Model
from keras.initializers import glorot_uniform

import tensorflow as tf
from keras import backend as K
K.set_image_data_format("channels_first")

num_cores = 4
GPU = 0
CPU = 1
if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)


col = 8
row = 40

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def max_min_normalization(data_array):
    rows = data_array.shape[0]
    cols = data_array.shape[1]

    temp_array = np.zeros((rows,cols))
    col_min = data_array.min(axis=0)
    col_max = data_array.max(axis=0)

    for i in range(0,rows,1):
        for j in range(0,cols,1):
            temp_array[i][j] = (data_array[i][j]-col_min[j])/(col_max[j]-col_min[j])
    return temp_array

#残差块：标准块、卷积块
# 1、identity block
def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base   = 'bn'  + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    # save the input value
    X_shortcut = X

    # first component of main path
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid',
               name=conv_name_base+'2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2a')(X)
    X = Activation('relu')(X)

    # second component of main path
    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same',
               name=conv_name_base+'2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid',
               name=conv_name_base+'2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2c')(X)

    # Final step
    # Add shortcut value to main path, and pass it through a ReLU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

# 2、convolutional block
def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the identity block

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, 主路径中间的那个CONV的窗口形状
    filters -- python整数列表, 定义主路径每个CONV层中的滤波器的数量
    stage --整数，用于命名层，取决于他们在网络中的位置          阶段
    block --字符串/字符，用于命名层，取决于他们在网络中的位置   块
    s -- 整数，指定滑动的大小

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base   = 'bn'  + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    # save the input value
    X_shortcut = X

    # first component of main path
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding='valid',
               name=conv_name_base+'2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2a')(X)
    X = Activation('relu')(X)

    # second component of main path
    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same',
               name=conv_name_base+'2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid',
               name=conv_name_base+'2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2c')(X)

    # shortcut path
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding='valid',
            name=conv_name_base+'1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base+'1')(X_shortcut)

    # Final step
    # Add shortcut value to main path, and pass it through a ReLU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape=(1,row,col), classes=8):

    X_input = Input(input_shape)
    # stage 1
    X = Conv2D(filters=4, kernel_size=(3,3), strides=(1,1), padding='valid',
               name='conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), strides=(2,2))(X)
    # stage 2
    X = convolutional_block(X, f=3, filters=[4,4,16], stage=2, block='a', s=1)
    X = identity_block(X, f=3, filters=[4,4,16], stage=2, block='b')
    X = identity_block(X, f=3, filters=[4,4,16], stage=2, block='c')

    # stage 3
    X = convolutional_block(X, f=3, filters=[8,8,32], stage=3, block='a', s=1)
    X = identity_block(X, f=3, filters=[8,8,32], stage=3, block='b')
    X = identity_block(X, f=3, filters=[8,8,32], stage=3, block='c')
    X = identity_block(X, f=3, filters=[8,8,32], stage=3, block='d')
    # stage 4
    X = convolutional_block(X, f=3, filters=[16,16,64], stage=4, block='a', s=1)
    X = identity_block(X, f=3, filters=[16,16,64], stage=4, block='b')
    X = identity_block(X, f=3, filters=[16,16,64], stage=4, block='c')
    X = identity_block(X, f=3, filters=[16,16,64], stage=4, block='d')
    X = identity_block(X, f=3, filters=[16,16,64], stage=4, block='e')
    X = identity_block(X, f=3, filters=[16,16,64], stage=4, block='f')
    # stage 5
    X = convolutional_block(X, f=3, filters=[16,16,64], stage=5, block='a', s=1)
    X = identity_block(X, f=3, filters=[16,16,64], stage=5, block='b')
    X = identity_block(X, f=3, filters=[16,16,64], stage=5, block='c')

    # fully connected output layer
    X = Flatten(name='flatten')(X)
    X = Dense(256,activation='relu',name='fc1')(X)
    X = Dense(classes, activation='softmax', name='fc2')(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet50_semg')
    return model

"下载数据和标签"
f = scio.loadmat('./emg_mat/gesture_emg_40.mat')
data  = f['data']
label = f['labels'][0]

"随机打乱数据和标签"
N = data.shape[0]
index = np.random.permutation(N)
data  = data[index,:]
label = label[index]

"对数据特征归一化"
data = max_min_normalization(data)

"将label的数据类型改成int,将label的数字都减1"
label = label.astype(int)
label = label - 1

"转换标签为one-hot"
label = label.reshape((1,label.shape[0]))
label = convert_to_one_hot(label,8)

"生成训练样本及标签、测试样本及标签"
num_train = int(round(N*0.8))
num_test  = N-num_train
train_data  = data[0:num_train,:]
test_data   = data[num_train:N,:]
train_label = label[0:num_train,:]
test_label  = label[num_train:N,:]

print("train data shape:",train_data.shape)
print("train label shape:",train_label.shape)
print("test data shape:",test_data.shape)
print("test label shape:",test_label.shape)

X_train = np.expand_dims(train_data.reshape((num_train,row,col)), axis=1)
Y_train = train_label
X_test  = np.expand_dims(test_data.reshape((num_test,row,col)), axis=1)
Y_test  = test_label

model = ResNet50(input_shape = (1,row,col), classes = 8)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=50, batch_size=200)


preds_train = model.evaluate(X_train, Y_train)
print("Train Loss = " + str(preds_train[0]))
print("Train Accuracy = " + str(preds_train[1]))

preds_test  = model.evaluate(X_test, Y_test)
print("Test Loss = " + str(preds_test[0]))
print("Test Accuracy = " + str(preds_test[1]))



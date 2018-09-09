#-*-coding:utf-*-
import scipy.io as scio
import numpy as np
from keras.layers import Input,Dense, Activation,Flatten, Conv2D, AveragePooling2D
from keras.models import Model
import keras.backend as K

K.set_image_data_format('channels_first')
K.set_learning_phase(1)
np.random.seed(1)
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

def CNN_semg(input_shape=(1,40,8), classes=8):

    X_input = Input(input_shape)
    "block 1"
    "32 filters,  a row of the length of number of electrodes,  ReLU"
    X = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),padding='same', name='conv1')(X_input)
    X = Activation('relu', name='relu1')(X)

    "block 2"
    "32 filters 3*3,  ReLU,  average pool 3*3"
    X = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),padding='same', name='conv2')(X)
    X = Activation('relu', name='relu2')(X)
    X = AveragePooling2D((3,3), strides=(2,2), name='pool1')(X)

    "block 3"
    "64 filters 5*5,  ReLu,  average pool 3*3"
    X = Conv2D(filters=64, kernel_size=(5,5), strides=(1,1),padding='same', name='conv3')(X)
    X = Activation('relu', name='relu3')(X)
    X = AveragePooling2D((3,3), strides=(1,1), name='pool2')(X)

    "block 4"
    "64 filters 5*1,  ReLU"
    X = Conv2D(filters=64, kernel_size=(5,1), strides=(1,1),padding='same', name='conv4')(X)
    X = Activation('relu', name='relu4')(X)

    "block 5"
    "filters 1*1,  softmax loss"
    #X = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1),padding='same', name='conv5')(X)
    X = Flatten(name='flatten')(X)

    #X = Dense(256,    activation='relu',    name='fc1')(X)
    X = Dense(classes, activation='softmax', name='fc2')(X)

    model = Model(inputs=X_input, outputs=X, name='CNN_semg')

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
print np.unique(label)
"转换标签为one-hot"
label = label.reshape((1,label.shape[0]))
label = convert_to_one_hot(label,8)

"生成训练样本及标签、测试样本及标签"
num_train = int(round(N*0.8))
num_test  = N-num_train
print num_train
train_data  = data[0:num_train,:]
test_data   = data[num_train:N,:]
train_label = label[0:num_train,:]
test_label  = label[num_train:N,:]

print("train data shape:",train_data.shape)
print("train label shape:",train_label.shape)
print("test data shape:",test_data.shape)
print("test label shape:",test_label.shape)

X_train = np.expand_dims(train_data.reshape((num_train,40,8)), axis=1)
Y_train = train_label
X_test  = np.expand_dims(test_data.reshape((num_test,40,8)), axis=1)
Y_test  = test_label

model = CNN_semg(input_shape = (1,40,8), classes = 8)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=100, batch_size=64)

preds_train = model.evaluate(X_train, Y_train)
print("Train Loss = " + str(preds_train[0]))
print("Train Accuracy = " + str(preds_train[1]))

preds_test  = model.evaluate(X_test, Y_test)
print("Test Loss = " + str(preds_test[0]))
print("Test Accuracy = " + str(preds_test[1]))

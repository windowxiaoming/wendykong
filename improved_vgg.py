#-*-coding:utf-8-*-
import scipy.io as scio
import numpy as np
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


def VGG16_semg(input_shape=(row,col,1), classes=8):
    from keras.layers import Input
    from keras.layers import Conv2D
    from keras.layers import BatchNormalization
    from keras.layers import AveragePooling2D
    from keras.layers import Flatten
    from keras.layers import Dense
    from keras.models import Model
    X_input = Input(input_shape)

    "block 1"
    X = Conv2D(filters=4, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block1_conv1')(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Conv2D(filters=4, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block1_conv2')(X)


    "block 2"
    X = Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block2_conv1')(X)
    X = BatchNormalization(axis=3)(X)
    X = Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block2_conv2')(X)
    X = BatchNormalization(axis=3)(X)

    "block 3"
    X = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block3_conv1')(X)
    X = BatchNormalization(axis=3)(X)
    X = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block3_conv2')(X)
    X = BatchNormalization(axis=3)(X)
    X = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block3_conv3')(X)
    X = BatchNormalization(axis=3)(X)
    X = AveragePooling2D((2,2), strides=(2,2), name='block3_pool')(X)

    "block 4"
    X = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block4_conv1')(X)
    X = BatchNormalization(axis=3)(X)
    X = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block4_conv2')(X)
    X = BatchNormalization(axis=3)(X)
    X = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block4_conv3')(X)
    X = BatchNormalization(axis=3)(X)

    "block 5"
    X = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block5_conv1')(X)
    X = BatchNormalization(axis=3)(X)    
    X = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block5_conv2')(X)
    X = BatchNormalization(axis=3)(X)
    X = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block5_conv3')(X)
    X = BatchNormalization(axis=3)(X)

    X = Flatten(name='flatten')(X)
    X = Dense(256,activation='relu',name='fc1')(X)
    X = Dense(classes, activation='softmax', name='fc2')(X)
    model = Model(inputs=X_input, outputs=X, name='VGG16_semg')
    return model

"下载数据和标签"
f = scio.loadmat('./emg_mat/gesture_emg_40.mat')
data  = f['data']
label = f['labels'][0]
print data.shape
print np.unique(label)
"随机打乱数据和标签"
N = data.shape[0]
index = np.random.permutation(N)
data  = data[index,:]
label = label[index]

"对数据特征归一化"
#data = max_min_normalization(data)

"将label的数据类型改成int,将label的数字都减1"
label = label.astype(int)
label = label - 1

label = convert_to_one_hot(label,8)

"生成训练样本及标签、测试样本及标签"
num_train = int(round(N*0.8))
num_test  = N-num_train
print num_train
train_data  = data[0:num_train,:]
test_data   = data[num_train:N,:]
train_label = label[0:num_train]
test_label  = label[num_train:N]

print("train data shape:",train_data.shape)
print("train label shape:",train_label.shape)
print("test data shape:",test_data.shape)
print("test label shape:",test_label.shape)

X_train = np.expand_dims(train_data.reshape((num_train,row,col)), axis=3)
Y_train = train_label
X_test  = np.expand_dims(test_data.reshape((num_test,row,col)), axis=3)
Y_test  = test_label

from keras.optimizers import SGD
model = VGG16_semg(input_shape = ( row,col, 1), classes = 8)
optimizers=SGD(lr=0.01, momentum=0.5, decay=0.0, nesterov=True)
model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=50, batch_size=200)


preds_train = model.evaluate(X_train, Y_train)
print("Train Loss = " + str(preds_train[0]))
print("Train Accuracy = " + str(preds_train[1]))

preds_test  = model.evaluate(X_test, Y_test)
print("Test Loss = " + str(preds_test[0]))
print("Test Accuracy = " + str(preds_test[1]))




import random
import numpy as np


import sys
from keras.models import Sequential
from keras.layers import LSTM, Dense
nbatch_size =32
nEpoches = 30
channelSize = 8
timesteps = 40
winSize = 40
num_classes = 8
from keras.utils.vis_utils import plot_model 
import scipy.io as scio
    
def buildlstm():
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,   input_shape=(channelSize, timesteps)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(16))  # return a single vector of dimension 32
    model.add(Dense(num_classes, activation='softmax'))
    plot_model(model, to_file='model_lstm.png',show_shapes=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print(model.summary())
    return  model
    pass

def runTrain(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train,  batch_size= nbatch_size, epochs= nEpoches,shuffle=True)
    score = model.evaluate(x_test, y_test, batch_size=nbatch_size)
    print('evaluate score:', score)
    pass        

def test():

    filepath="./emg_mat/gesture_emg_40.mat"
    f = scio.loadmat(filepath)
    data  = f['data']
    label = f['labels'][0]    
    y = label
    x = data

    total_num = y.shape[0]
    train_num = int(total_num * 0.9)
    test_num = int(total_num * 0.1)

    num_list = [i for i in range(0, total_num)]
    random.shuffle(num_list)

    train_indexs_list = num_list[:train_num]
    train_data = np.zeros((0,channelSize,winSize))
    train_labs = np.zeros((train_num, 8))


    test_indexs_list = num_list[train_num:]
    test_data = np.zeros((0,channelSize,winSize))
    test_labs = np.zeros((test_num, 8))


    for i in range(train_num):
        train_index = train_indexs_list[i]
        train_datatmp = x[train_index].reshape((1,channelSize,winSize))
        train_data = np.concatenate((train_data,train_datatmp),0)
        train_labs[i, y[train_index]-1] = 1
        i += 1

    for j in range(test_num):
        test_index = test_indexs_list[j]
        test_datatmp = x[test_index].reshape((1,channelSize,winSize))
        test_data = np.concatenate((test_data,test_datatmp),0)
        test_labs[j, y[test_index]-1] = 1
        j += 1

    model = buildlstm()
    runTrain(model, train_data, test_data, train_labs, test_labs)
    pass


if __name__ == "__main__":
    sys_code_type = sys.getfilesystemencoding()
    test()


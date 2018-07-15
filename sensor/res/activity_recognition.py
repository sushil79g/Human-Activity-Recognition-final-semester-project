from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import optimizers
import datetime
from functools import lru_cache


@lru_cache()
def main():
    random_seed = 611
    np.random.seed(random_seed)
    plt.style.use('ggplot')

    def readData(filePath):
        # attributes of the dataset
        columnNames = ['user_id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
        data = pd.read_csv(filePath, header=None, names=columnNames, na_values=';', low_memory=False)
        # pprint(data)
        return data

    # def featureNormalize(dataset):
    #     mu = np.mean(dataset, axis=0)
    #     sigma = np.std(dataset, axis=0)
    #     return (dataset - mu) / sigma
    # #
    # def plotAxis(axis, x, y, title):
    #     axis.plot(x, y)
    #     axis.set_title(title)
    #     axis.xaxis.set_visible(False)
    #     axis.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    #     axis.set_xlim([min(x), max(x)])
    #     axis.grid(True)

    #
    # def plotActivity(activity, data):
    #     fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    #     plotAxis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    #     plotAxis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    #     plotAxis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    #     plt.subplots_adjust(hspace=0.2)
    #     fig.suptitle(activity)
    #     plt.subplots_adjust(top=0.9)
    #     plt.show()

    #
    #
    def windows(data, size):
        start = 0
        while start < data.count():
            yield int(start), int(start + size)
            start += (size / 2)

    def segment_signal(data, window_size=90):
        segments = np.empty((0, window_size, 3))
        labels = np.empty((0))
        for (start, end) in windows(data['timestamp'], window_size):
            x = data['x-axis'][start:end]
            y = data['y-axis'][start:end]
            z = data['z-axis'][start:end]
            if (len(data['timestamp'][start:end]) == window_size):
                segments = np.vstack([segments, np.dstack([x, y, z])])
                labels = np.append(labels, stats.mode(data['activity'][start:end])[0][0])
        return segments, labels

    #
    dataset = readData("H:/projects/8thseme/src/sensor/res/tracker.txt")
    # for activity in np.unique(dataset['activity']):
    #     subset = dataset[dataset['activity'] == activity][:180]
    # print(activity)
    # pprint(subset)
    # plotActivity(activity, subset)
    segments, labels = segment_signal(dataset)
    labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
    numOfRows = segments.shape[1]
    numOfColumns = segments.shape[2]
    numChannels = 1
    numFilters = 128  # number of filters in Conv2D layer
    # kernal size of the Conv2D layer
    kernalSize1 = 2
    # max pooling window size
    poolingWindowSz = 2
    # number of filters in fully connected layers
    numNueronsFCL1 = 128
    numNueronsFCL2 = 128
    # split ratio for test and validation
    trainSplitRatio = 0.8
    # number of epochs
    Epochs = 10
    # batchsize
    batchSize = 10
    # number of total clases
    numClasses = labels.shape[1]
    # dropout ratio for dropout layer
    dropOutRatio = 0.2
    # reshaping the data for network input
    reshapedSegments = segments.reshape(segments.shape[0], numOfRows, numOfColumns, 1)
    # splitting in training and testing data
    trainSplit = np.random.rand(len(reshapedSegments)) < trainSplitRatio
    trainX = reshapedSegments[trainSplit]
    testX = reshapedSegments[~trainSplit]
    trainX = np.nan_to_num(trainX)
    testX = np.nan_to_num(testX)
    trainY = labels[trainSplit]
    testY = labels[~trainSplit]

    def cnnModel():
        model = Sequential()
        # adding the first convolutionial layer with 32 filters and 5 by 5 kernal size, using the rectifier as the activation function
        model.add(
            Conv2D(numFilters, (kernalSize1, kernalSize1), input_shape=(numOfRows, numOfColumns, 1), activation='relu'))
        # adding a maxpooling layer
        model.add(MaxPooling2D(pool_size=(poolingWindowSz, poolingWindowSz), padding='valid'))
        # adding a dropout layer for the regularization and avoiding over fitting
        model.add(Dropout(dropOutRatio))
        # flattening the output in order to apply the fully connected layer
        model.add(Flatten())
        # adding first fully connected layer with 256 outputs
        model.add(Dense(numNueronsFCL1, activation='relu'))
        # adding second fully connected layer 128 outputs
        model.add(Dense(numNueronsFCL2, activation='relu'))
        # adding softmax layer for the classification
        model.add(Dense(numClasses, activation='softmax'))
        # Compiling the model to generate a model
        adam = optimizers.Adam(lr=0.001, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        for layer in model.layers:
            print(layer.name)
        model.fit(trainX, trainY, validation_split=1 - trainSplitRatio, epochs=10, batch_size=batchSize, verbose=2)
        score = model.evaluate(testX, testY, verbose=2)
        print('Baseline Error: %.2f%%' % (100 - score[1] * 100))
        return model

    model = cnnModel()
    return model
    # x = 2
    # while x > 1:
    #     prediction = model.predict(inputdata.reshape(-1, 90, 3, 1))
    #     labe = ['downstair', 'jogging', 'sitting', 'standing', 'upstair', 'walking']
    #     result = labe[np.argmax(prediction)]
    #     print(result)

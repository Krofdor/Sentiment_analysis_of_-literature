import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb


def vectorize(sequenses, dimension=10000):
    results = np.zeros((len(sequenses), dimension))
    for i, sequenses in enumerate(sequenses):
        results[i, sequenses] = 1
    return results



(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
data = vectorize(data)
targets = np.array(targets).astype("float32")
test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]
butch_size = 500

#train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
model = Sequential()
#model.add(Dense(50, activation="relu", input_shape=(10000, )))
#model.add(Dropout(0.3, noise_shape=None, seed=None))
#model.add(Dense(50, activation="relu"))
#model.add(Dropout(0.2, noise_shape=None, seed=None))
#model.add(Dense(50, activation="relu"))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))
#model.summary()

model.add(Embedding(10000, 128))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    #class_mode="binary",
    #metrics=["accuracy"]
)

#X = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
results = model.fit(
    train_x, train_y,
    nb_epoch=1,
    #epochs=2,
    batch_size=2,
    #validation_data=(test_x, test_y)
)
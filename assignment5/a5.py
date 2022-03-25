import tensorflow as tf
from tensorflow import keras
import pickle
#from stop_words import get_stop_words
import statistics
import numpy as np


with open(file="keras-data.pickle", mode="rb") as file:
    data = pickle.load(file)

# y-data needs to be in numpy array
x_train, y_train = data["x_train"], np.array(data["y_train"])
x_test, y_test = data["x_test"], np.array(data["y_test"])


vocab_size = data["vocab_size"]
max_length = data["max_length"]

lenghts = []

for i in x_train:
    lenghts.append(len(i))

avg = sum(lenghts)/len(lenghts)
median = statistics.median(lenghts)

LEN = round(avg)

print("median : ", median)
print("Average : ", round(avg))

x_train = keras.preprocessing.sequence.pad_sequences(
    x_train, maxlen=LEN, dtype='int32', padding="post")

x_test = keras.preprocessing.sequence.pad_sequences(
    x_test, maxlen=LEN, dtype='int32', padding="post")

embed = 32
hidden = 16
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(
    vocab_size, embed, input_length=LEN))

model.add(tf.keras.layers.LSTM(hidden))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

print(x_train.shape, x_train.dtype)
print(y_test.shape, y_test.dtype)

model.compile(optimizer="Adam", loss="binary_crossentropy",
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=1)


loss, acc = model.evaluate(x_test, y_test, verbose=1)

print("Accuracy is: ", acc)
print("Loss is: ", loss)
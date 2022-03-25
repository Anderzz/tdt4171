import numpy as np
import tensorflow as tf
import pickle
import statistics

#load data
with open(file="keras-data.pickle", mode="rb") as file:
    data = pickle.load(file)

x_train = data['x_train']
y_train = np.array(data['y_train'])
x_test = data['x_test']
y_test = np.array(data['y_test'])

vocab_size = data["vocab_size"]
max_len = data["max_length"]

lenghts = []

for i in x_train:
    lenghts.append(len(i))

avg = sum(lenghts)/len(lenghts)
median = statistics.median(lenghts)

LEN = round(avg)

print("max : ", max_len)
print("Average : ", round(avg))

#preprocessing
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=LEN, padding="post")
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=LEN, padding="post")
print(x_train.shape, y_test.shape)
#define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 32, input_length=LEN))
model.add(tf.keras.layers.LSTM(16, activation='tanh', dropout=0.2,recurrent_dropout=0.0))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1)
model.evaluate(x_test, y_test, verbose=0)

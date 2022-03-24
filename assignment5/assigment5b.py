import numpy as np
import tensorflow as tf
import pickle

#load data
with open(file="keras-data.pickle", mode="rb") as file:
    data = pickle.load(file)

x_train = data['x_train']
y_train = np.array(data['y_train'])
x_test = data['x_test']
y_test = np.array(data['y_test'])

vocab_size = data["vocab_size"]
max_len = data["max_length"]

#preprocessing
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len, padding="post")
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len, padding="post")
print(x_train.shape, x_test.shape)
#define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 32, input_length=max_len))
model.add(tf.keras.layers.LSTM(16,activation='tanh',dropout=0.0,recurrent_dropout=0.0))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1)
model.evaluate(x_test, y_test)

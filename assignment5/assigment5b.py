from tabnanny import verbose
import numpy as np
import tensorflow as tf
import pickle

#load data
with open(file="keras-data.pickle", mode="rb") as file:
    data = pickle.load(file)

#check if running on gpu
if tf.config.list_physical_devices('GPU'):
    print("Running on GPU")

x_train = data['x_train']
y_train = np.array(data['y_train'])
x_test = data['x_test']
y_test = np.array(data['y_test'])

vocab_size = data["vocab_size"]
max_len = data["max_length"]

lengths = [len(i) for i in x_train]
avg = round(sum(lengths) / len(lengths))

#preprocessing
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=avg)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=avg)
in_len = x_train.shape[1]
#define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 64, input_length=in_len))
model.add(tf.keras.layers.LSTM(32, activation='tanh', recurrent_activation='sigmoid', dropout=0.2))
model.add(tf.keras.layers.Dense(16,activation="tanh"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
model.compile(optimizer="Adamax", loss="binary_crossentropy", metrics=['accuracy', 'AUC'])
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=True)
model.evaluate(x_test, y_test, verbose=True)


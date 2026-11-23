"""Experiment 23: Implementing an Attention Mechanism (basic custom model).
A custom Bahdanau-style (additive) attention layer on top of a BiLSTM for IMDB sentiment."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

VOCAB_SIZE, MAX_LEN = 10000, 200


class BahdanauAttention(layers.Layer):
    """score_t = v^T tanh(W h_t);  alpha = softmax(score);  context = sum_t alpha_t * h_t"""

    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units, activation="tanh")
        self.V = layers.Dense(1)

    def call(self, hidden_states):                       # (batch, time, features)
        score = self.V(self.W(hidden_states))            # (batch, time, 1)
        weights = tf.nn.softmax(score, axis=1)           # attention over time steps
        context = tf.reduce_sum(weights * hidden_states, axis=1)  # (batch, features)
        return context, tf.squeeze(weights, -1)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
x_train = tf.keras.utils.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = tf.keras.utils.pad_sequences(x_test, maxlen=MAX_LEN)

inputs = layers.Input(shape=(MAX_LEN,))
x = layers.Embedding(VOCAB_SIZE, 64)(inputs)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
context, attn_weights = BahdanauAttention(64)(x)
x = layers.Dropout(0.5)(context)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
print("Test [loss, accuracy]:", model.evaluate(x_test, y_test, verbose=0))

# Inspect which words the model attends to for one review
attn_model = models.Model(inputs, attn_weights)
word_index = tf.keras.datasets.imdb.get_word_index()
inv = {v + 3: k for k, v in word_index.items()}          # IMDB index offset = 3
sample = x_test[:1]
w = attn_model.predict(sample)[0]
tokens = [inv.get(i, "") for i in sample[0]]
top = np.argsort(w)[-10:][::-1]
print("Top-attended words:", [(tokens[i], round(float(w[i]), 4)) for i in top if tokens[i]])

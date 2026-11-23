"""Experiment 22: Sequence Classification using RNN/LSTM (Keras/TensorFlow).
Task: IMDB movie-review sentiment (binary) with a Bidirectional LSTM."""
import tensorflow as tf
from tensorflow.keras import layers, models

VOCAB_SIZE, MAX_LEN = 10000, 200

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
x_train = tf.keras.utils.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = tf.keras.utils.pad_sequences(x_test, maxlen=MAX_LEN)

model = models.Sequential([
    layers.Embedding(VOCAB_SIZE, 64),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.build(input_shape=(None, MAX_LEN))
model.summary()

model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {loss:.4f} | Test accuracy: {acc:.4f}")

# Predict on a few test reviews
probs = model.predict(x_test[:5]).ravel()
for p, y in zip(probs, y_test[:5]):
    print(f"pred={p:.3f} ({'positive' if p > 0.5 else 'negative'}) | actual={'positive' if y else 'negative'}")

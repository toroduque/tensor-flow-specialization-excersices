# Load the data from the sarcasm.json and parse it
import json
import tensorflow as tf
import numpy as np

with open('./tmp/sarcasm.json', 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# Hyperparameters
oov_tok = '<OOV>'
embedding_dim = 16
trunc_type = 'post'
padding_type = 'post'
max_length = 100
vocab_size = 3000
training_size = 20000

# Split sentences between training data and testing data
training_sentences = np.array(sentences[0:training_size])
training_labels = np.array(labels[0:training_size])

testing_sentences = np.array(sentences[training_size:])
testing_labels = np.array(labels[training_size:])

# Tokenize all the words from the sentences using the Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(
    sequences, 
    padding=padding_type, 
    truncating=trunc_type, 
    maxlen=max_length
)

test_tokenizer = Tokenizer(num_words=vocab_size)
test_tokenizer.fit_on_texts(testing_sentences)
test_sequences = tokenizer.texts_to_sequences(testing_sentences)
test_padded = pad_sequences(
    test_sequences, 
    padding=padding_type, 
    truncating=trunc_type, 
    maxlen=max_length
)

# Define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)

history = model.fit(
    padded,
    training_labels,
    epochs=30,
    validation_data=(test_padded, testing_labels),
    verbose=2
)

import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

# plot_graphs(history, 'acc')
plot_graphs(history, 'loss')
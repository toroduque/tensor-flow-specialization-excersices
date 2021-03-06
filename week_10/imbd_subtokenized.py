import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

imbd, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
train_data, test_data = imbd['train'], imbd['test']

# Fix to solve the error: 
# "sigmoid_cross_entropy_with_logits raise ValueError("logits and labels must have the same shape (%s vs %s)" % "
# More info here: https://github.com/huggingface/tokenizers/issues/253
train_data = train_data.map(lambda x_text, x_label: (x_text, tf.expand_dims(x_label, -1)))
test_data = test_data.map(lambda x_text, x_label: (x_text, tf.expand_dims(x_label, -1)))

# Subwords Tokenizer
tokenizer = info.features['text'].encoder

# Split data between training and testing data
BUFFER_SIZE = 10000
BATCH_SIZE = 64

# train_dataset = train_data.shuffle(BUFFER_SIZE)
# train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
# test_dataset = test_data.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_data))

embedding_dim = 64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train model
num_epochs = 10
model.fit(
    train_data,
    epochs=num_epochs,
    validation_data=test_data
)



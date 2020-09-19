import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()

song_lyrics = "I took her out, it was a Friday night \n I wore cologne to get the feeling right \n We started making out and she took off my pants \n But then I turned on the TV \n And that's about the time she walked away from me \n Nobody likes you when you're twenty three \n And I'm still more amused by TV shows \n What the hell is A.D.D.? \n My friends say I should act my age \n What's my age again, what's my age again? \n Then later on, on the drive home \n I called her mom from a pay phone \n I said I was the cops \n And your husband's in jail \n The state looks down on sodomy \n And that's about the time that bitch hung up on me \n And I'm still more amused by prank phone calls \n What the hell is call ID? \n And that's about the time she walked away from me \n Nobody likes you when you're twenty three \n And you still act like you're in freshman year \n What the hell is wrong with me? \n That's about the time she broke up with me \n No one should take themselves so seriously \n With many years ahead to fall in line \n Why would you wish that on me? \n I never want to act my age"

corpus = song_lyrics.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Find the length of the longest sentence in the corpus
max_sequence_len = max([len(x) for x in input_sequences])

# Pad all of the sequences so they'll have the same length
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# X = Take all characters but the last of each sentence
# Y = the last character of each sentence
xs = input_sequences[:,:-1]
labels = input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Hyperparameters
emb_dim = 64


# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(total_words, emb_dim, input_length=max_sequence_len-1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(xs,ys, epochs=500, verbose=1)

import matplotlib.pyplot as plt

def plot_graphs(history,string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history,'loss')

seed_text = "I took her out"
next_words = 100

for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
	predicted = model.predict_classes(token_list, verbose=0)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word

print(seed_text)






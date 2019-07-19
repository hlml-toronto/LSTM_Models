# The file contains the functions to design and train the LSTM model on the dataset as well as generate new lyrics
# by Eugene Klyshko

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
import warnings; warnings.simplefilter('ignore')

import keras.models as km
import keras.layers as kl
from keras.optimizers import RMSprop
import random



def build_model(num_of_lstm_units = 128, sentence_length, num_words):
	''' The function builds an LSTM model. Needs a bit of tuning. So far the most optimal
	'''
	print('Building network.')
	model = km.Sequential()
	model.add(kl.LSTM(num_of_lstm_units, input_shape = (sentence_length, num_words)))
	model.add(Dropout(0.1))
	model.add(kl.Dense(num_words, activation = 'softmax'))
	optimizer = RMSprop(lr=0.01) # 'sgd'
	model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

	return model


def fit_model(model, name):
	'''The function fits the given model for N epochs and save it to the file models/model_watever.h5'''
	# Fit!  Begin elevator music...
	print('Beginning fit.')
	epochs = 20
	fit = model.fit(x, y, epochs = 20, batch_size = 256)

	# Save the model so that we can use it as a starting point.
	model.save('models/model_{}.h5'.format(name))
	print('Model saved.')
	return model


def generate(modelfile, desired_length = 500, sentence_length, num_words, decoding):
	'''The function takes in the trained model and generates the desired length of text.'''
	# Get the model.
	print('Reading model file.')
	model = km.load_model(modelfile)

	# Get the meta-data.

	# Randomly choose ? words from the dictionary of words as our
	# starting sentence.
	seed = []
	for i in range(sentence_length):
	    seed.append(decoding[random.randint(0, num_words - 1)])


	# Encode the seed sentence.
	x = np.zeros((1, sentence_length, num_words), dtype = np.bool)
	for i, w in enumerate(seed):
	    x[0, i, encoding[w]] = 1

	text = ''

	# Run the seed sentence through the model.  Add the output to the
	# generated text.  Take the output and append it to the seed sentence
	# and remove the first word from the seed sentence.  Then repeat until
	# you've generated as many words as you like.
	for i in range(desired_length):

	    # Get the most-probably next word.
	    pred = np.argmax(model.predict(x, verbose = 0))

	    # Add it to the generated text.
	    text += decoding[pred] + ' '

	    # Encode the next word.
	    next_word = np.zeros((1, 1, num_words), dtype = np.bool)
	    next_word[0, 0, pred] = 1

	    # Concatenate the next word to the seed sentence, but leave off
	    # the first element so that the length stays the same.
	    x = np.concatenate((x[:, 1:, :], next_word), axis = 1)
   
	# Print out the generated text.
	print("Generated lyrics: \n")
	print(text)


# data = pd.read_csv('data/lyrics_titles_AutoPump.csv')
# sentence_length = 25
# num_words = 2500

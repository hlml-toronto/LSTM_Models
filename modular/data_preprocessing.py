# The file contains the functions to preprocess data frame with lyrics in order to build the corpus of words and training data
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



def dframe2txt(data, filename):
	'''This function takes in dataframe of lyrics and titles and saves it into txt file separating lyrics and titles by stars'''
	all_lyrics = open(filename, 'w')
	for row in data.itertuples():
	    text = row.lyrics
	    title = row.title
	    if text.startswith('     '):
        	continue
	    all_lyrics.write(title + '\n')
	    all_lyrics.write('********************\n')
	    all_lyrics.write(text + '\n')
	    all_lyrics.write('********************\n\n\n')
	all_lyrics.close()


def clean_lines(file, cleaned_file):
	'''The function reads in a txt file with formatted lyrics/titles and makes the text cleaner - removing unnecessary information, 
	making all lowercase, etc. You can add many possible cleaning procedures. It then saves the text into another cleaned_file'''
	lines = []
	all_lyrics = open(file, 'r')
	for line in all_lyrics:
	    if line.startswith(' '):
	        line = line.lstrip()
	    line = line.lower()
	    if line.startswith('[chorus'):
	        lines.append('[chorus] \n')
	    elif line.startswith('[verse'):
	        lines.append('[verse] \n')
	    elif line.startswith('[hook'):
	        lines.append('[hook] \n')
	    elif line.startswith('[intro'):
	        lines.append('[intro] \n')
	    elif line.startswith('[outro'):
	        lines.append('[outro] \n')
	    elif line.startswith('[bridge'):
	        lines.append('[bridge] \n')
	    elif line.startswith('[interlude'):
	        lines.append('[interlude] \n')
	    elif line.startswith('lyrics for this'):
	        lines.append('\n')
	    else:
	        lines.append(line)
	all_lyrics.close()
	all_lyrics_cleaned = open(cleaned_file, 'w')
	for line in lines:
    	all_lyrics_cleaned.write(line)
	all_lyrics_cleaned.close() 


def create_corpus(cleaned_file):
	'''This functions reads in a cleaned version of the txt file and splits it into tokens. Here you decide if you want to
	include punctiona, new lines, etc. as tokens. Should be very specific tune up for your project'''
	
	# Read in the entire file.
	f = open(cleaned_file)
	corpus0 = f.read()
	f.close()

	# Separate the punctuation from the words, so that words with
	# punctuation do not get counted as distinct from words without
	# punctuation.  Same for new line characters.

	corpus0 = corpus0.replace(',', ' ,')
	corpus0 = corpus0.replace('(', ' ( ')
	corpus0 = corpus0.replace(')', ' ) ')
	#corpus0 = corpus0.replace('[', ' [ ')
	#corpus0 = corpus0.replace(']', ' ] ')
	corpus0 = corpus0.replace('.', ' . ')
	corpus0 = corpus0.replace(';', ' ; ')
	corpus0 = corpus0.replace(':', ' : ')
	corpus0 = corpus0.replace('!', ' ! ')
	corpus0 = corpus0.replace('?', ' ? ')
	corpus0 = corpus0.replace('********************', ' ******************** ')
	#corpus0 = corpus0.replace('*', ' * ')
	corpus0 = corpus0.replace("’", '\'')
	corpus0 = corpus0.replace("\'\'", ' " ')
	corpus0 = corpus0.replace('"', ' " ')
	corpus0 = corpus0.replace('\r\n', ' \r\n ')

	# Separate the dashes from any words they're attached to.
	corpus0 = corpus0.replace('-', ' - ')
	corpus0 = corpus0.replace('\n', ' \n ')

	# Convert the text to lower case.
	corpus0 = corpus0.lower()

	# Split the words by spaces;
	corpus1 = corpus0.split(' ')

	while (corpus1.count('') > 0): 
	    corpus1.remove('')
	    
	print('Length of corpus is now ', len(corpus1))
	return corpus1

def vocabulary(corpus):
	'''The function creates a sorted list of unique words from the corpus, encode and decode them with position number.
	This step is important for the x, y creation. Returns words, encoding, decoding dictionaries'''

	# Preprocessing is done.  Now get the unique words, and encode them.
	words = sorted(list(set(corpus)))
	num_words = len(words)
	encoding = {w: i for i, w in enumerate(words)}
	decoding = {i: w for i, w in enumerate(words)}

	print('We have', num_words, 'different words.')
	return (words, encoding, decoding)


def lstm_dataset(corpus, sentence_length = 50):
	'''This function creates x, y for training purposes'''
	words, encoding, decoding = vocabulary(corpus)
	# Chop up the data into x and y, slice into roughly num_chars
	# overlapping 'sentences' of length sentence_length.  Encode the
	# characters.
	x_data = []
	y_data = []
	for i in range(0, len(corpus) - sentence_length):
	    sentence = corpus[i: i + sentence_length]
	    next_word = corpus[i + sentence_length]
	    x_data.append([encoding[word] for word in sentence])
	    y_data.append(encoding[next_word])

	# good word: phronesis
	num_sentences = len(x_data)
	print('We have', len(x_data), 'sentences.')

	# Create the variables to hold the data as it will be used.
	x = np.zeros((num_sentences, sentence_length, num_words), dtype = np.bool)
	y = np.zeros((num_sentences, num_words), dtype = np.bool)

	# Populate the sentences.
	print('Encoding data.')
	for i, sentence in enumerate(x_data):
	    for t, encoded_word in enumerate(sentence):
	        x[i, t, encoded_word] = 1
	    y[i, y_data[i]] = 1

	# The processing of the data takes a fair amount of time.  Save
	# the data so we don't have to do this again.  We do this in a
	# numpy file since the data is large and the shelve can't handle
	# it.

	print('Saving processed data.')
	np.save('x.npy', x)
	np.save('y.npy', y)

	return x, y
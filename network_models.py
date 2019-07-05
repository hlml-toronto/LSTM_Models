import numpy as np
import os

import keras.models as km
import keras.layers as kl


def recipe_lstm(num_lstms, sentence_length, num_words):

    print( 'Building network.' )
    model = km.Sequential()
    model.add(kl.LSTM(num_lstms, input_shape = (sentence_length, num_words)))
    model.add(kl.Dense(num_words, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop',
                  metrics = ['accuracy'])

    return model


def unet_lstm():

    input_data = kl.Input(shape = (np.shape(x_train)[1],), name='input')

    x = kl.Dense(30, name = 'hidden'+str(1), activation = 'tanh')(input_data)
    output = kl.Dense(1, name='output', activation='sigmoid')(x)
    model = km.Model(inputs = input_data, outputs = output)
    model.compile(optimizer = 'sgd', metrics = ['accuracy'], loss = 'binary_crossentropy')

    return model

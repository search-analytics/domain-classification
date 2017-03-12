'''
Same as example script but this incorporates preliminary Web of Science data

Data is available on Search Analytics machine: 

Discussion around script available here: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
Performance sucks after 2 epochs (~10% accuracy)
'''

from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import sys

basedir = os.path.abspath(os.path.dirname(__file__))

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
VALIDATION_SPLIT = .2
EMBEDDING_DIM = 100

texts = []
labels = []
labels_index = {}
with open(os.path.join(basedir, "data", "abstracts.txt"), "r") as f:
	texts = f.readlines()

print('Found %s texts.' % len(texts))

with open(os.path.join(basedir, "data", "labels.txt"), "r") as f:
	word_labels = f.readlines()
	for name in word_labels:
		label_id = len(labels_index)
		labels_index[name] = label_id
		labels.append(label_id)

# print (texts[0:10])
# print (len(labels))
# print (labels_index)

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


# prepare embedding matrix
# nb_words = min(MAX_NB_WORDS, len(word_index))
# embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
# for word, i in word_index.items():
#     if i >= MAX_NB_WORDS:
#         continue
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
# embedding_layer = Embedding(nb_words,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=False)


embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index) + 1, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print (x_train.shape)
print (y_train.shape)
print (x_val.shape)
print (y_val.shape)

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=2, batch_size=128)


# Orig
# ==================================================================================

# batch_size = 32
# nb_epoch = 2
# max_words = 1000 #in each doc

# print('Loading data...')
# (X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)
# print(len(X_train), 'train sequences')
# print(len(X_test), 'test sequences')

# print (y_test[:10])
# print (y_train[:10])

# nb_classes = np.max(y_train) + 1
# print(nb_classes, 'classes')

# print('Vectorizing sequence data...')
# tokenizer = Tokenizer(nb_words=max_words)
# X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
# X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
# print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)

# print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)
# print('Y_train shape:', Y_train.shape)
# print('Y_test shape:', Y_test.shape)

# print('Building model...')
# model = Sequential()
# model.add(Dense(512, input_shape=(max_words,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# history = model.fit(X_train, Y_train,
#                     nb_epoch=nb_epoch, batch_size=batch_size,
#                     verbose=1, validation_split=0.1)

# # score = model.evaluate(X_test, Y_test,
# #                        batch_size=batch_size, verbose=1)

# prediction = model.predict(X_test, batch_size=batch_size, verbose=1)

# print (len(X_test[0]))
# print (prediction[0])


# print('Test score:', score[0])
# print('Test accuracy:', score[1])
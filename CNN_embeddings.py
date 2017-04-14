'''
Same as example script but this incorporates preliminary Web of Science data

Data is available on Search Analytics machine: 

Discussion around script available here: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
'''

from __future__ import print_function
import os
import numpy as np
import json
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import keras
import sys
import argparse
from gensim.models.word2vec import Word2Vec


# python3 CNN_embeddings.py -a "single-category-abstracts.txt" -l "single-category-labels.txt"

parser = argparse.ArgumentParser(description='Train classifier to identify domain from text')
parser.add_argument('-a', '--abstracts', help='file containing abstracts for training/testing (one per line)', default="abstracts.txt")
parser.add_argument('-l', '--labels', help='file containing labels for abstracts (one per line)', default="labels.txt")
args = parser.parse_args()

basedir = os.path.abspath(os.path.dirname(__file__))

MAX_NB_WORDS = 40000
MAX_SEQUENCE_LENGTH = 1500
VALIDATION_SPLIT = .2
EMBEDDING_DIM = 100

texts = []
labels = []
labels_index = {}
indices = [] #when only using samplefrom non-cs categories

cs_labels = ["Computer Science, Software Engineering", "Computer Science, Cybernetics", "Computer Science, Hardware & Architecture", 
"Computer Science, Information Systems", "Computer Science, Theory & Methods", "Computer Science, Artificial Intelligence", 
"Computer Science, Interdisciplinary Applications"]


########################################################
# Only considering CS domains, all else is "other"
########################################################

with open(os.path.join(basedir, "data", args.labels), "r") as f:
	word_labels = f.readlines()
	
	other_samples = {}

	labels_index["Other"] = 0 

	for i, name in enumerate(word_labels):
		if name.replace("\n","") in cs_labels:
			
			if not name in labels_index:
				label_id = len(labels_index)
				labels_index[name] = label_id
			
			labels.append(labels_index[name])
			indices.append(i)

		else:
			if not name in other_samples or other_samples[name] < 10:

				labels.append(0)
				indices.append(i)

				if name in other_samples:
					other_samples[name] += 1
				else:
					other_samples[name] = 1

with open(os.path.join(basedir, "data", args.abstracts), "r") as f:
	all_texts = f.readlines()

	for i, abstract in enumerate(all_texts):
		if i in indices:
			texts.append(abstract)

print(json.dumps(labels_index, indent=4))
print('Using %s texts.' % len(texts))

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

print (len(sequences))
print (sequences[0:2])

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
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print (x_train.shape)
print (y_train.shape)
print (x_val.shape)
print (y_val.shape)

# tensorboard --logdir=/Users/hundman/documents/data_science/search-analytics/domain-classification/logs
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

# happy learning!
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=25, batch_size=128, callbacks=[tbCallBack])

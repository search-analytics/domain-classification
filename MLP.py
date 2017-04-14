'''
Simple NN (MLP)

Data is available on Search Analytics machine: 
'''

from __future__ import print_function
import os
import numpy as np
from numpy import zeros
import json
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
# from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import keras
import sys
import argparse

from multiprocessing import cpu_count
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split

# python3 LSTM-w2v.py -a "single-category-abstracts.txt" -l "single-category-labels.txt"

parser = argparse.ArgumentParser(description='Train classifier to identify domain from text')
parser.add_argument('-a', '--abstracts', help='max number of categories allowed to be present for an abstract', default="abstracts.txt")
parser.add_argument('-l', '--labels', help='max number of categories allowed to be present for an abstract', default="labels.txt")
args = parser.parse_args()

basedir = os.path.abspath(os.path.dirname(__file__))

##############################################################################
# Preprocessing
##############################################################################

texts = []
labels = []
labels_index = {}
indices = [] #when only using samples from non-cs categories

cs_labels = ["Computer Science, Software Engineering", "Computer Science, Cybernetics", "Computer Science, Hardware & Architecture", 
"Computer Science, Information Systems", "Computer Science, Theory & Methods", "Computer Science, Artificial Intelligence", 
"Computer Science, Interdisciplinary Applications"]

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

##############################################################################
# Vectorization
##############################################################################

MAX_NB_WORDS = 40000
VALIDATION_SPLIT = .2
BATCH_SIZE = 32
num_classes = len(labels_index)
num_features = 700 # Word2Vec number of features
document_max_num_words = 500  #Limit to a fixed number of words

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
x = tokenizer.texts_to_matrix(texts, mode='tfidf')

x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=VALIDATION_SPLIT)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

##############################################################################
# Train
##############################################################################

model = Sequential()
    
model.add(Dense(128, input_shape=(MAX_NB_WORDS,)))
model.add(Activation('relu'))
model.add(Dropout(0.7))

# http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
# model.add(Dense(512, input_shape=(1024,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))

# model.add(Dense(256, input_shape=(512,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(len(labels_index)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                nb_epoch=5, batch_size=BATCH_SIZE,
                verbose=1, validation_split=0.2)

# model.save(os.path.join(basedir, MODEL_NAME))

# Evaluate model
score, acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    
print('Score: %1.4f' % score)
print('Accuracy: %1.4f' % acc)


# tensorboard --logdir=/Users/hundman/documents/data_science/search-analytics/domain-classification/logs
# tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

# happy learning!
# history = model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=25, batch_size=128, callbacks=[tbCallBack])



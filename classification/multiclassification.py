from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import lda
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
import dill
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold


# ========================= global variables =========================

dir='/Users/asitangm/Desktop/SA/data_wos/'


# ========================= read the data =========================

texts=open(dir+'abstracts_cs.txt','r')
labels=open(dir+'labels_cs.txt','r')


texts = texts.readlines()
labels=labels.readlines()


texts=[line.strip() for line in texts]
labels=[[line.strip()] for line in labels]

le = preprocessing.LabelEncoder()
le.fit(labels)

# labels=[x.split('\t') for x in labels]


texts_test=open(dir+'abstracts_cs_test.txt','r')
labels_test=open(dir+'labels_cs_test.txt','r')

texts_test = texts_test.readlines()
labels_test=labels_test.readlines()


texts_test=[line.strip() for line in texts_test]
labels_test=[line.strip() for line in labels_test]

texts_all=texts_test+texts

print len(texts_all)

labels_test=[x.split('\t') for x in labels_test]

# ========================= preprocess =========================

multilabelbinarizer=MultiLabelBinarizer()

labels=multilabelbinarizer.fit_transform(labels)
print labels.shape


labels_test=multilabelbinarizer.transform(labels_test)
print labels_test.shape

# ========================= vectorize =========================

dictvectorizer = DictVectorizer(sparse=False)

countvectorizer=CountVectorizer()
tfidftransformer=TfidfTransformer()

texts_all=countvectorizer.fit_transform(texts_all)
# tfidftransformer.fit(texts_all)



texts=countvectorizer.transform(texts)
# texts=tfidftransformer.transform(texts)
X=texts
y=labels


texts_test=countvectorizer.transform(texts_test)
# texts_test=tfidftransformer.transform(texts_test)
X_test=texts_test
y_test=labels_test

print X.shape
print y.shape

print X_test.shape
print y_test.shape

# ========================= prepare classifiers =========================



clasif2=RandomForestClassifier()
clasif3=MLPClassifier(max_iter=1000,learning_rate='adaptive')
clasif=VotingClassifier(estimators=[('rf', clasif2), ('nn', clasif3)], voting='soft')

multi_target_classif = OneVsRestClassifier(clasif)
skf = KFold(n_splits=5)



# ========================= LDA =========================

topicnum=20
model = lda.LDA(n_topics=topicnum, n_iter=500, random_state=1)

model.fit(texts_all)
print 'shape all',model.doc_topic_.shape

docs=model.transform(texts,max_iter=100)
print 'shape train',docs.shape

featurearray=[]
for i in range(0,docs.shape[0]):
    doc_topics={'Topic-'+str(y):x for x,y in zip(docs[i],range(0,topicnum))}
    featurearray.append(doc_topics)

dictvector=dictvectorizer.fit_transform(featurearray)
X=dictvector

dictvectorizer = DictVectorizer(sparse=False)
docs=model.transform(texts_test,max_iter=100)
print 'testshape',docs.shape

featurearray=[]
for i in range(0,docs.shape[0]):
    doc_topics={'Topic-'+str(y):x for x,y in zip(docs[i],range(0,topicnum))}
    featurearray.append(doc_topics)

dictvector=dictvectorizer.fit_transform(featurearray)
X_test=dictvector

print X.shape


# ========================= Run classification =========================



multi_target_classif.fit(X,y)
y_pred=multi_target_classif.predict_proba(X_test)
print average_precision_score(y_test, y_pred,average=None)




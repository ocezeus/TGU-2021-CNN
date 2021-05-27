#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('C:\\Users\\biscu\\anaconda3\\lib\\site-packages\\')
import pandas as pd
import numpy as np
import jieba
import jieba.posseg as pseg
import re
import string
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from gensim.models import word2vec
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import load_model
from keras.models import Sequential
import matplotlib.pyplot as plt


train_data = pd.read_csv('newTrain.csv', lineterminator='\n') #读数据
test_data = pd.read_csv('newTest.csv', lineterminator='\n')

#预处理
def encodeLabel(data):
    listLable=[]
    for lable in data['lable']:
        listLable.append(lable)
    le = LabelEncoder() #规格化
    resultLable=le.fit_transform(listLable)
    return resultLable

trainLable = encodeLabel(train_data)
testLable = encodeLabel(test_data)
print(testLable)

def getReview(data):
    listReview=[]
    for review in data['review']:
        listReview.append(review)
    return listReview

trainReview = getReview(train_data)
testReview = getReview(test_data)
print(testReview)

#分词
def stopwordslist():
    stopwords = [line.strip() for line in open('中文停用词表.txt',encoding='UTF-8').readlines()]
    return stopwords

def deleteStop(sentence):
    stopwords=stopwordslist()
    outstr=""
    for i in sentence:
        if i not in stopwords and i!="\n":
            outstr+=i
    return outstr
def wordCut(Review):
    Mat=[]
    for rec in Review:
        seten=[]
        rec = re.sub('[%s]' % re.escape(string.punctuation), '',rec)
        fenci=jieba.lcut(rec)
        stc=deleteStop(fenci)
        seg_list=pseg.cut(stc)                                              #标注词性
        for word,flag in seg_list:
            if flag not in ["nr","ns","nt","nz","m","f","ul","l","r","t"]:  #去掉词性
                seten.append(word)
        Mat.append(seten)
    return Mat

trainCut = wordCut(trainReview)
testCut = wordCut(testReview)
wordCut = trainCut+testCut
print(testCut)

fileDic=open('wordCut.txt','w',encoding='UTF-8')
for i in wordCut:
    fileDic.write(" ".join(i))
    fileDic.write('\n')
fileDic.close()
words = [line.strip().split(" ") for line in open('wordCut.txt',encoding='UTF-8').readlines()]

maxLen=100

#word2vec训练
num_featrues = 100
min_word_count = 3
num_workers =4
context = 4 #上下文窗口

model = word2vec.Word2Vec(wordCut, workers=num_workers, size=num_featrues, min_count=min_word_count,window=context)
model.init_sims(replace=True) # 强制归一化
model.save("mgcModel")
model.wv.save_word2vec_format("CNNmgc",binary=False)
print(model)

w2v_model = word2vec.Word2Vec.load("mgcModel")

tokenizer=Tokenizer()
tokenizer.fit_on_texts(words)
vocab = tokenizer.word_index
print(vocab)

trainID = tokenizer.texts_to_sequences(trainCut) #词频编号
testID = tokenizer.texts_to_sequences(testCut)

trainSeq=pad_sequences(trainID,maxlen=maxLen) #规格化
testSeq=pad_sequences(testID,maxlen=maxLen)

trainCate = to_categorical(trainLable, num_classes=2)  #独热
testCate= to_categorical(testLable, num_classes=2)

#word2vec替换
embedding_matrix = np.zeros((len(vocab) + 1, 100))
for word, i in vocab.items():
    try:
        embedding_vector = w2v_model[str(word)]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        continue

#训练
main_input = Input(shape=(maxLen,), dtype='float64')
embedder = Embedding(len(vocab) + 1, 100, input_length=maxLen, weights=[embedding_matrix], trainable=False)
model=Sequential()
model.add(embedder)
model.add(Conv1D(256,3,padding='same',activation='relu'))
model.add(MaxPool1D(maxLen-5,3,padding='same'))
model.add(Conv1D(32,3,padding='same',activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=2,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(trainSeq, trainCate, batch_size=256, epochs=6,validation_split=0.2)
model.save("TextCNN")

#预测与评估
mainModel = load_model('TextCNN')
result = mainModel.predict(testSeq)  # 预测样本属于每个类别的概率
print(result)
print(np.argmax(result,axis=1))
score = mainModel.evaluate(testSeq,
                           testCate,
                           batch_size=32)
print(score)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Valid'],loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Valid'],loc='upper left')
plt.show()



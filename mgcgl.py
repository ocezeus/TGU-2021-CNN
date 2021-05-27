#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('C:\\Users\\biscu\\anaconda3\\lib\\site-packages\\')
import numpy as np
import jieba
import jieba.posseg as pseg
import re
import string
import pymysql
import time
from keras.preprocessing.text import Tokenizer
from gensim.models import word2vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model



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


while True:
    db = pymysql.connect(host="localhost",
                         user="root",
                         password="123456",
                         db="deliciousfoods",
                         port=3306)
    cursor1 = db.cursor()
    cursor2 = db.cursor()
    sqlin = "select content from message_info order by id desc limit 1"
    cursor1.execute(sqlin)
    test_data = cursor1.fetchall()

    print(test_data[0][0])
    wordCut1 = wordCut(test_data[0])
    print(wordCut1)

    fileDic=open('mgc.txt','w',encoding='UTF-8')
    for i in wordCut1:
        fileDic.write(" ".join(i))
        fileDic.write('\n')
    fileDic.close()
    words = [line.strip().split(" ") for line in open('mgc.txt',encoding='UTF-8').readlines()]

    maxLen=100

    w2v_model = word2vec.Word2Vec.load("CNNw2vModel")

    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(words)
    vocab = tokenizer.word_index
    testID = tokenizer.texts_to_sequences(wordCut1)
    testSeq=pad_sequences(testID,maxlen=maxLen)

    embedding_matrix = np.zeros((len(vocab) + 1, 100))
    for word, i in vocab.items():
        try:
            embedding_vector = w2v_model[str(word)]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            continue

    mainModel = load_model('TextCNN')
    result = mainModel.predict(testSeq)  # 预测样本属于每个类别的概率
    print(result)
    resu = np.argmax(result,axis=1)
    print(resu)

    if resu == 0:
        print('未检测到敏感词')
    else:
        print('检测到敏感词，开冲')
        sqlout = "update message_info set content='检测到敏感词，已屏蔽' order by id desc limit 1"
        cursor2.execute(sqlout)
        db.commit()
        print('螺旋冲锋')

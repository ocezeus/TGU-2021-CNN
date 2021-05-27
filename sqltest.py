import pymysql
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

db= pymysql.connect(host="localhost",
                    user="root",
                    password="123456",
                    db="deliciousfoods",
                    port=3306)
cursor = db.cursor()
sql = "select content from message_info order by id desc limit 1"
cursor.execute(sql)

results = cursor.fetchall()

print(results[0][0])




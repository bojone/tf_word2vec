#! -*- coding:utf-8 -*-
#Keras版的Word2Vec，作者：苏剑林，http://kexue.fm
#Keras 2.0.6 ＋ Tensorflow 测试通过

import numpy as np
from keras.layers import Input,Embedding,Lambda
from keras.models import Model
import keras.backend as K

word_size = 128 #词向量维度
window = 5 #窗口大小
nb_negative = 16 #随机负采样的样本数
min_count = 10 #频数少于min_count的词将会被抛弃
nb_worker = 4 #读取数据的并发数
nb_epoch = 2 #迭代次数，由于使用了adam，迭代次数1～2次效果就相当不错
subsample_t = 1e-5 #词频大于subsample_t的词语，会被降采样，这是提高速度和词向量质量的有效方案
nb_sentence_per_batch = 20
#目前是以句子为单位作为batch，多少个句子作为一个batch（这样才容易估计训练过程中的steps参数，另外注意，样本数是正比于字数的。）

import pymongo
class Sentences: #语料生成器，必须这样写才是可重复使用的
    def __init__(self):
        self.db = pymongo.MongoClient().weixin.text_articles
    def __iter__(self):
        for t in self.db.find(no_cursor_timeout=True).limit(100000):
            yield t['words'] #返回分词后的结果

sentences = Sentences()
words = {} #词频表
nb_sentence = 0 #总句子数
total = 0. #总词频

for d in sentences:
    nb_sentence += 1
    for w in d:
        if w not in words:
            words[w] = 0
        words[w] += 1
        total += 1
    if nb_sentence % 10000 == 0:
        print u'已经找到%s篇文章'%nb_sentence

words = {i:j for i,j in words.items() if j >= min_count} #截断词频
id2word = {i+1:j for i,j in enumerate(words)} #id到词语的映射，0表示UNK
word2id = {j:i for i,j in id2word.items()} #词语到id的映射
nb_word = len(words)+1 #总词数（算上填充符号0）

subsamples = {i:j/total for i,j in words.items() if j/total > subsample_t}
subsamples = {i:subsample_t/j+(subsample_t/j)**0.5 for i,j in subsamples.items()} #这个降采样公式，是按照word2vec的源码来的
subsamples = {word2id[i]:j for i,j in subsamples.items() if j < 1.} #降采样表

def data_generator(): #数据生成器
    while True:
        x,y = [],[]
        _ = 0
        for d in sentences:
            d = [0]*window + [word2id[w] for w in d if w in word2id] + [0]*window
            r = np.random.random(len(d))
            for i in range(window, len(d)-window):
                if d[i] in subsamples and r[i] > subsamples[d[i]]: #满足降采样条件的直接跳过
                    continue
                x.append(d[i-window:i]+d[i+1:i+1+window])
                y.append([d[i]])
            _ += 1
            if _ == nb_sentence_per_batch:
                x,y = np.array(x),np.array(y)
                z = np.zeros((len(x), 1))
                yield [x,y],z
                x,y = [],[]
                _ = 0

input_words = Input(shape=(window*2,), dtype='int32')
input_vecs = Embedding(nb_word, word_size, name='word2vec')(input_words)
input_vecs_sum = Lambda(lambda x: K.sum(x, axis=1))(input_vecs) #CBOW模型，直接将上下文词向量求和

target_word = Input(shape=(1,), dtype='int32')
negatives = Lambda(lambda x: K.random_uniform((K.shape(x)[0], nb_negative), 0, nb_word, 'int32'))(target_word)
samples = Lambda(lambda x: K.concatenate(x))([target_word,negatives]) #构造抽样，负样本随机抽。负样本也可能抽到正样本，但概率小。

softmax_weights = Embedding(nb_word, word_size, name='W')(samples)
softmax_biases = Embedding(nb_word, 1, name='b')(samples)
softmax = Lambda(lambda x: 
                    K.softmax((K.batch_dot(x[0], K.expand_dims(x[1],2))+x[2])[:,:,0])
                )([softmax_weights,input_vecs_sum,softmax_biases]) #用Embedding层存参数，用K后端实现矩阵乘法，以此复现Dense层的功能

#留意到，我们构造抽样时，把目标放在了第一位，也就是说，softmax的目标id总是0，这可以从data_generator中的z变量的写法可以看出

model = Model(inputs=[input_words,target_word], outputs=softmax)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(data_generator(), 
                    steps_per_epoch=nb_sentence/nb_sentence_per_batch, 
                    epochs=nb_epoch,
                    workers=nb_worker,
                    use_multiprocessing=True
                   )

model.save_weights('word2vec.model')

#通过词语相似度，检查我们的词向量是不是靠谱的
embeddings = model.get_weights()[0]
normalized_embeddings = embeddings / (embeddings**2).sum(axis=1).reshape((-1,1))**0.5

def most_similar(w):
    v = normalized_embeddings[word2id[w]]
    sims = np.dot(normalized_embeddings, v)
    sort = sims.argsort()[::-1]
    sort = sort[sort > 0]
    return [(id2word[i],sims[i]) for i in sort[:10]]

import pandas as pd
pd.Series(most_similar(u'科学'))

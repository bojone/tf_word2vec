#! -*- coding:utf-8 -*-

from collections import defaultdict
import numpy as np
import datetime
import pickle


class Word2Vec:
    def __init__(self, train_data=None, word_size=128, window=5, min_count=5, model='cbow', shared_softmax=True, nb_negative=16, epochs=2, batch_size=8000):
        if train_data: #如果有数据传入，那么启动建模和训练程序；如果没有，那么只初始化一个空对象。
            import tensorflow as tf
            global tf
            self.train_data = train_data
            self.word_size = word_size
            self.window = window
            self.min_count = min_count
            self.model = model
            self.shared_softmax = shared_softmax
            self.nb_negative = nb_negative
            self.epochs = epochs
            self.batch_size = batch_size
            self.words = defaultdict(int)
            self.word_count()
            self.build_model()
            self.train_model()
    def word_count(self): #统计词频，过滤低频词
        for total,t in enumerate(self.train_data):
            for w in t:
                self.words[w] += 1
            if total % 10000 == 0:
                print '%s, get %s articles, %s uique words.'%(datetime.datetime.now(), total, len(self.words))
        self.words = {i:j for i,j in self.words.items() if j >= self.min_count}
        self.id2word = {i+1:j for i,j in enumerate(self.words.keys())}
        self.word2id = {j:i for i,j in self.id2word.items()}
        self.total_sentences = total + 1
        self.total_words = len(self.word2id)
        self.total_word_frequency = sum(self.words.values())
        print '%s, min_count=%s left %s unique words.'%(datetime.datetime.now(), self.min_count, self.total_words)
    def random_softmax_loss(self, nb_negative, inputs, targets, weights, biases=None): #定义loss函数：随机抽样、点乘、交叉熵。
        nb_classes, real_batch_size = tf.shape(weights)[0], tf.shape(targets)[0]
        negative_sample = tf.random_uniform([real_batch_size, nb_negative], 0, nb_classes, dtype=tf.int32)
        random_sample = tf.concat([targets, negative_sample], axis=1)
        sampled_weights = tf.nn.embedding_lookup(weights, random_sample)
        if biases:
            sampled_biases = tf.nn.embedding_lookup(biases, random_sample)
            sampled_logits = tf.matmul(inputs, sampled_weights, transpose_b=True)[:,0,:] + sampled_biases
        else:
            sampled_logits = tf.matmul(inputs, sampled_weights, transpose_b=True)[:,0,:]
        sampled_labels = tf.zeros([real_batch_size], dtype=tf.int32)
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sampled_logits, labels=sampled_labels))
    def build_model(self): #建立模型，比较常规
        self.sess = tf.Session()
        self.embeddings = tf.Variable(tf.random_uniform([self.total_words+1, self.word_size], -0.05, 0.05))
        self.normalized_embeddings = tf.nn.l2_normalize(self.embeddings, 1)
        self.target_words = tf.placeholder(tf.int32, shape=[None, 1])
        if self.model == 'cbow':
            self.input_words = tf.placeholder(tf.int32, shape=[None, 2*self.window])
            self.input_vecs = tf.nn.embedding_lookup(self.embeddings, self.input_words)
            self.input_vecs = tf.expand_dims(tf.reduce_sum(self.input_vecs, 1), 1)
        else:
            self.input_words = tf.placeholder(tf.int32, shape=[None, 1])
            self.input_vecs = tf.nn.embedding_lookup(self.embeddings, self.input_words)
        if self.shared_softmax:
            self.loss = self.random_softmax_loss(self.nb_negative, self.input_vecs, self.target_words, self.embeddings)
        else:
            self.softmax_weights = tf.Variable(tf.random_uniform([self.total_words+1, self.word_size], -0.05, 0.05))
            self.softmax_biases = tf.Variable(tf.zeros([self.total_words+1]))
            self.loss = self.random_softmax_loss(self.nb_negative, self.input_vecs, self.target_words, self.softmax_weights, self.softmax_biases)
    def data_generator(self): #数据生成器，用来生成每个batch
        if self.model == 'cbow':
            x,y = [],[]
            for idx,t in enumerate(self.train_data):
                t = [self.word2id[i] for i in t if i in self.word2id]
                for i,s in enumerate(t):
                    win = t[max(0, i-self.window): i] + t[i+1: min(len(t), i+self.window)]
                    win += [0]*(2*self.window-len(win))
                    x.append(win)
                    y.append([s])
                    if len(x) == self.batch_size:
                        yield idx, np.array(x), np.array(y)
                        x,y = [],[]
            if x:
                yield idx, np.array(x), np.array(y)
        else:
            x,y = [],[]
            for idx,t in enumerate(self.train_data):
                t = [self.word2id[i] for i in t if i in self.word2id]
                for i,s in enumerate(t):
                    win = t[max(0, i-self.window): i] + t[i+1: min(len(t), i+self.window)]
                    for w in win:
                        x.append([w])
                        y.append([s])
                        if len(x) == self.batch_size:
                            yield idx, np.array(x), np.array(y)
                            x,y = [],[]
            if x:
                yield idx, np.array(x), np.array(y)
    def train_model(self): #模型训练部分，主要使用了多进程，一个进程负责生成数据，一个负责训练数据
        from multiprocessing import Process,Queue
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        for e in range(self.epochs):
            queue = Queue(1000)
            def put_into_queue(queue):
                data = self.data_generator()
                for d in data:
                    while queue.full():
                        pass
                    queue.put(d)
            putting = Process(target=put_into_queue, args=(queue,))
            putting.start()
            count = 0
            while True:
                idx,x,y = queue.get()
                if count%100 == 0:
                    loss_ = self.sess.run(self.loss, feed_dict={self.input_words: x, self.target_words: y})
                    print '%s, epoch %s, trained on %s articles, loss: %s'%(datetime.datetime.now(), e+1, idx, loss_)
                self.sess.run(self.train_step, feed_dict={self.input_words: x, self.target_words: y})
                count += 1
                if idx+1 == self.total_sentences:
                    break
            putting.terminate()
        self.embeddings = self.sess.run(self.embeddings) #训练完成后，重新初始化变量，使得在没有tf的环境中也能够调用模型
        self.normalized_embeddings = self.sess.run(self.normalized_embeddings)
        if not self.shared_softmax:
            self.softmax_weights = self.sess.run(self.softmax_weights)
            self.softmax_biases = self.sess.run(self.softmax_biases)
    def __getitem__(self, w):
        return self.embeddings[self.word2id[w]]
    def most_similar(self, word, topn=10): #通过cos相似度找近义词
        word_vec = self.normalized_embeddings[self.word2id[word]]
        word_sim = np.dot(self.normalized_embeddings[1:], word_vec)
        word_sim_argsort = word_sim.argsort()[::-1]
        return [(self.id2word[i+1], word_sim[i]) for i in word_sim_argsort[1:topn+1]]
    def log_proba(self, input_words): #由输入词预测词的概率分布对数
        input_words = [self.embeddings[self.word2id[w]] for w in input_words if w in self.words]
        if input_words:
            if self.model == 'cbow': #对于cbow模型，直接将输入词向量求和然后点乘并softmax，这是很自然的。
                input_words = sum(input_words)
                if self.shared_softmax:
                    logits = np.dot(self.embeddings, input_words)
                else:
                    logits = np.dot(self.softmax_weights, input_words) + self.softmax_biases
                logits = np.exp(logits - logits.max())+1e-12
                log_proba = np.log(logits/logits.sum())
            else: #对于skip-gram模型，则利用贝叶斯公式和特征独立假设来算
                logs = []
                for v in input_words:
                    if self.shared_softmax:
                        logits = np.dot(self.embeddings, v)
                    else:
                        logits = np.dot(self.softmax_weights, v) + self.softmax_biases
                    logits = np.exp(logits - logits.max())+1e-12
                    logs.append(np.log(logits/logits.sum()))
                log_proba = sum(logs) - np.array([0]+[(len(logs)-1)*(np.log(self.words[self.id2word[i]])-np.log(self.total_word_frequency)) for i in range(1,len(self.words)+1)])
            log_proba_argsort = log_proba.argsort()[::-1]
            return [(self.id2word[i], log_proba[i]) for i in log_proba_argsort if i != 0]
    def save_model(self, saved_folder): #保存模型，模型不可再训练（事实上再训练对于多数人来说没有意义）
        saved_folder = (saved_folder+'/').replace('//', '/')
        pickle.dump([self.words, self.word2id, self.id2word, self.model, self.shared_softmax], open(saved_folder+'words.pickle', 'w'))
        np.save(saved_folder+'embeddings', self.embeddings)
        if not self.shared_softmax:
            np.save(saved_folder+'softmax_weights', self.softmax_weights)
            np.save(saved_folder+'softmax_biases', self.softmax_biases)
    def load_model(self, saved_folder): #重新加载模型，但不可再训练
        saved_folder = (saved_folder+'/').replace('//', '/')
        self.words, self.word2id, self.id2word, self.model, self.shared_softmax = pickle.load(open(saved_folder+'words.pickle'))
        self.embeddings = np.load(saved_folder+'embeddings.npy')
        self.normalized_embeddings = self.embeddings/(self.embeddings**2).sum(axis=1).reshape((-1,1))**0.5
        if not self.shared_softmax:
            self.softmax_weights = np.load(saved_folder+'softmax_weights.npy')
            self.softmax_biases = np.save(saved_folder+'softmax_biases.npy')

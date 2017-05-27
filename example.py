from Word2Vec import *
import pymongo
db = pymongo.MongoClient().travel.articles
class texts:
    def __iter__(self):
        for t in db.find().limit(30000):
            yield t['words']

wv = Word2Vec(texts(), model='cbow', nb_negative=16, shared_softmax=True, epochs=2) #建立并训练模型
wv.save_model('myvec') #保存到当前目录下的myvec文件夹

#训练完成后可以这样调用
wv = Word2Vec() #建立空模型
wv.load_model('myvec') #从当前目录下的myvec文件夹加载模型

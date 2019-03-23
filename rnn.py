import get_feature
import numpy as np
import sklearn
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import SimpleRNN
from sklearn.model_selection import StratifiedKFold
from keras.layers.recurrent import LSTM
#import theano

# 读取dga和normal的csv文件
def get_X_y(dgacsv_path,normalcsv_path):
    dga = pd.read_csv(dgacsv_path)
    normal = pd.read_csv(normalcsv_path)

    X = list(dga['dga']) + list(normal['normal'])
    y = list(dga['labels']) + list(normal['labels'])
    
    # 去掉顶级域名
    # X = [x.split('.')[0] for x in X] 
    # print("split COM success!")

    return X,y

# 构造lstm模型
def build_LSTM_model(max_features, maxlen):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')
    return model

# 构造rnn模型
def build_rnn_model(max_features, maxlen,embedding_value,rnn_value):
    model = Sequential()
    model.add(Embedding(max_features, embedding_value, input_length=maxlen))#将maxlen维向量使用one-hot方法展开成max_features维向量，然后进行降维到128
    model.add(SimpleRNN(rnn_value))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')
    model.summary()
    return model

# run
def run(dgacsv_path, normalcsv_path,
        model_save_path='rnn_model.h5',
        nfolds=10, batch_size=32,epoch=1,
        embedding_value=128,rnn_value=128):

    X,y = get_X_y(dgacsv_path,normalcsv_path)
    
    # model的input_length，所有域名里数量占有99.9%的域名的长度界限
    maxlen = 38 

    # 读取一元文法的词典
    with open("tran.txt",'r') as file:
        lines =  file.readlines()
        valid_chars={}
        for line in lines:
            line = "".join(line.split("\n"))
            line = line.split(" ")       
            valid_chars[line[0]]=int(line[1])

    # 词典长度，model中input_dim
    max_features = len(valid_chars) + 1  
    XX = X
    
    # 将X里的域名按每个字符使用词典进行编码，如['google',...]->[[2,5,5,2,45,6],...]
    X = [[valid_chars[y] for y in x] for x in X]
     
    # # 追加5种特征值
    # X = get_feature.get_features(XX,X)
    # maxlen = 38+5
    
    # 按照maxlen补0并转为矩阵，如max_len=8, [[2,5,5,2,45,6],...]->[[0 0 2 5 5 2 45 6] [..]...]
    X = sequence.pad_sequences(X,maxlen=maxlen,dtype="double")
     
    f_data = []
    part = len(X)//nfolds
    
    # 模型建立
    model = build_rnn_model(max_features, maxlen,embedding_value=embedding_value,rnn_value=rnn_value)
    
    # 交叉验证：将数据集平均分为nfolds份，训练nfolds次，每一次训练在数据集中按顺序取一份作为测试集，剩下的作为训练集
    for fold in range(nfolds):
        
        if fold ==0:
            X_test = X[:(fold+1)*part]
            y_test = y[:(fold+1)*part]
            X_train = X[(fold+1)*part:]
            y_train = y[(fold+1)*part:]
        elif fold != nfolds-1 and fold!= 0:
            X_test = X[fold*part:(fold+1)*part]
            y_test = y[fold*part:(fold+1)*part]
            X_train = np.vstack((X[0:fold*part],X[(fold+1)*part:])) #纵向拼接矩阵
            y_train = y[0:fold*part]+y[(fold+1)*part:]
        elif fold == nfolds-1:
            X_test = X[fold*part:]
            y_test = y[fold*part:]
            X_train = X[0:fold*part]
            y_train = y[0:fold*part]
       
        print("fold %u/%u" % (fold + 1, nfolds))  # 输出当前fold
        
        # 每次训练batch_size个数据后更新一次权重。     
        model.fit(X_train, y_train, batch_size=batch_size,epochs=epoch)


        #已有的model在load权重过后
        #取某一层的输出为输出新建为model，采用函数模型
        # dense1_layer_model = Model(inputs=model.input,
        #                                     outputs=model.get_layer('dropout_1').output)
        # #以这个model的预测值作为输出
        # dense1_output = dense1_layer_model.predict(X_test)
        
        # print (dense1_output.shape)
        # print (dense1_output[0])
        # print (dense1_output[0][0])
  
        # 预测测试集集，返回被预测的标签概率
        probs = model.predict_proba(X_test)   
        
        #返回[[c00 c01][c10 c11]],c00:normal被判成normal的数量，c01:normal被判成dga的数量，c10:dga被判成normal的数量，c11：dga被判成dga的数量
        o_result = sklearn.metrics.confusion_matrix(y_test,probs>0.5,labels=[0,1]).ravel() 
        
        #评估训练集，得到方差值
        #a = model.evaluate(X_train,y_train,batch_size=32) 

        f_data.append({"fold":fold+1,"o_result":o_result})

    model.save(model_save_path)
    return f_data


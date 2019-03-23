import sys
import os
import get_feature
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn import metrics
from keras.preprocessing import sequence
def predict(csv_file_path,model_load_path):
    model = load_model(model_load_path)

    with open("tran.txt",'r') as file:
        lines =  file.readlines()
        valid_chars={}
        for line in lines:
            line = "".join(line.split("\n"))
            line = line.split(" ")       
            valid_chars[line[0]]=int(line[1])

    predict_data = pd.read_csv(csv_file_path)
    X_predict = list(predict_data['dga'])
    y_predict = list(predict_data['labels'])
    
    XX = X_predict
    # 去掉顶级域名
#     X_predict = [x.split('.')[0] for x in X_predict] 
#     print("split COM success!")

    max_features = len(valid_chars) + 1  # 字典长度，model中input_dim
    maxlen = 38
    X_predict = [[valid_chars[y] for y in x] for x in X_predict]  # 将X里的域名按每个字符进行编码，如['google',...]->[[2,5,5,2,45,6],...]
    
    # 追加5种特征值
#     X_predict = get_feature.get_features(XX,X_predict)
#     maxlen = 38+5
    
    X_predict = sequence.pad_sequences(X_predict,maxlen=maxlen)

    probs = model.predict_proba(X_predict)  # 预测测试集集，返回被预测的标签概率
    o_result = metrics.confusion_matrix(y_predict,probs>0.5,labels=[0,1]).ravel()

    return o_result

def markdown(o_result,file_name,rate,rate_csv_save_path):
    csv = pd.read_csv(rate_csv_save_path)
    p = pd.DataFrame()
    p["class"] = list(csv["class"])+[file_name]
    p["o_result"]=list(csv["o_result"])+[o_result]
    p["acc"]=list(csv["acc"])+[rate["acc"]]
    p["pre"]=list(csv["pre"])+[rate["pre"]]
    p["recall"] = list(csv["recall"])+[rate["recall"]]
    p["tpr"]=list(csv["tpr"])+[rate["tpr"]]
    p["fpr"]=list(csv["fpr"])+[rate["fpr"]]
    p.to_csv(rate_csv_save_path,columns=["class","o_result","acc","pre","recall","tpr","fpr"],index=False)

def get_rate(o_result):
    tn = o_result[0] # 预测为负实际为负
    fp = o_result[1] # 预测为正实际为负
    fn = o_result[2] # 预测为负实际为正
    tp = o_result[3] # 预测为正实际为正

    acc = (tp + tn) / (tp + fp + tn + fn)  # 准确度
    pre = tp / (tp + fp)  # 精确度
    recall = tp / (tp + fn)  # 召回率
    tpr = tp / (tp + fn)  # 真正例率、灵敏度
    fpr = fp / (fp + tn) # 假正例率

    rate = {"acc":acc,"pre":pre,"recall":recall,"tpr":tpr,"fpr":fpr}
    return rate

def main():
    sys.setrecursionlimit(10000)
    
    predict_docu_path = "data/each-dga"
    rate_csv_save_path = "ROC/each-model/rnn_model.csv"#"rnn128_embeddingX/predict/each_rnn_predict_1d4_rnn128_embedding160.csv"
    model_load_path ="ROC/each-model/rnn_model.h5"#"rnn128_embeddingX/rnn128_50w_result_embeding160.h5"
    
    files = os.listdir(predict_docu_path)
    p = pd.DataFrame()
    p["class"] = []
    p["o_result"]=[]
    p["acc"]=[]
    p["pre"]=[]
    p["recall"] = []
    p["tpr"]=[]
    p["fpr"]=[]
    p.to_csv(rate_csv_save_path,index=False,columns=["class","o_result","acc","pre","recall","tpr","fpr"])


    for file_name in files:
        csv_file_path = predict_docu_path+"/"+file_name
        o_result = predict(csv_file_path,model_load_path)
        rate = get_rate(o_result)
        markdown(o_result,file_name,rate,rate_csv_save_path)





main()
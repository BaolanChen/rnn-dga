import rnn
import numpy as np
import pandas as pd

# 记录交叉验证每轮结果
def write_file(f_data,train_mark_path):
    with open(train_mark_path, "w") as file:
        for d in f_data:
            file.write("fold:" + str(d["fold"]) + "\n")
            file.write("o_result:" + str(d["o_result"]) + "\n")
            #file.write("error:" + str(d["error"]) + "\n")
            file.write("-----------------------------" + "\n")
        file.close()

# 计算评测率
def get_average_rate(f_data,rate_csv_save_path):
    tn_l=[]
    fp_l=[]
    fn_l=[]
    tp_l=[]
    for i in f_data:
        tn_l.append(i["o_result"][0])
        fp_l.append(i["o_result"][1])
        fn_l.append(i["o_result"][2])
        tp_l.append(i["o_result"][3])

    tn = np.mean(tn_l)
    fp = np.mean(fp_l)
    fn = np.mean(fn_l)
    tp = np.mean(tp_l)

    acc = (tp + tn) / (tp + fp + tn + fn)  # 准确度
    pre = tp / (tp + fp)  # 精确度
    recall = tp / (tp + fn)  # 召回率
    tpr = tp / (tp + fn)  # 真正率、灵敏度
    fpr = fp / (fp + tn)

    p = pd.DataFrame()
    p["acc"] = [acc]
    p["pre"] = [pre]
    p["recall"] = [recall]
    p["tpr"] = [tpr]
    p["fpr"] = [fpr]
    p.to_csv(rate_csv_save_path,columns=["acc","pre","recall","tpr","fpr"],index=False)


def main():

    dga_path = "data/all/dga_mixture_1d2.csv"
    normal_path = "data/all/normal_mixture_1d2.csv"
    model_save_path="rnn128_embeddingX/100w_rnn128_embeding128_addfeatures.h5"
    rate_csv_save_path = "rnn128_embeddingX/100w_rnn128_embeding128_addfeatures.csv"
    train_mark_path = "rnn128_embeddingX/100w_rnn128_embeding128_addfeatures.txt"

    # f_data = rnn.run(dga_path, normal_path,model_save_path=model_save_path,nfolds=10,epoch=1,
    #                 embedding_value=96,rnn_value=128)

    # write_file(f_data,train_mark_path)
    # get_average_rate(f_data,rate_csv_save_path)

    f_data = rnn.run(dga_path, normal_path,model_save_path=model_save_path,nfolds=10,epoch=1,
                    embedding_value=128,rnn_value=128)

    write_file(f_data,train_mark_path)
    get_average_rate(f_data,rate_csv_save_path)


    


main()
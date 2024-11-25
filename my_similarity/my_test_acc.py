import csv
import pandas as pd


def write_simi(file, simi):
    with open(file, "r") as f:
        result = f.readlines()
        bert_simi_list = [i.strip("\n") for i in result]

        file_path = 'similarity_data_defined.csv'

        # 读取CSV文件到DataFrame
        df = pd.read_csv(file_path)

        # 检查新数据长度是否与DataFrame长度一致
        if len(bert_simi_list) != len(df):
            raise ValueError("新数据长度与CSV文件中的行数不一致")

        # 将新数据写入到'bert'这一列
        df[simi] = bert_simi_list

        # 将DataFrame写回CSV文件
        df.to_csv(file_path, index=False)

        print("数据已成功写入{}列".format(simi))

write_simi("simcse_similarity_result.txt", "simcse")
write_simi("bert_similarity_result.txt", "bert")

file_path = 'similarity_data_defined.csv'
# 读取CSV文件到DataFrame
df = pd.read_csv(file_path)

label_list = df["label"]

bert_list = df["bert"]
bert_count = 0
for i, j in zip(label_list, bert_list):
    if i == 1 and j >= 0.75 or i == 2 and j < 0.75:
        bert_count +=1
    #print("----i-----" + str(i) +  "-----bert------", str(j))
bert_acc = bert_count / len(bert_list)
print("bert的准确度为：", bert_acc)

simcse_list = df["simcse"]

simcse_count = 0
for i, j in zip(label_list, simcse_list):
    if i == 1 and j >= 0.75 or i == 2 and j < 0.75:
        simcse_count +=1
    #print("----i-----" + str(i) +  "-----bert------", str(j))
simcse_count = simcse_count / len(simcse_list)
print("simcse的准确度为：", simcse_count)





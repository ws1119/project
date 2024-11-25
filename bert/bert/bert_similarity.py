import csv
import random

import pandas as pd
import torch
# from pytorch_transformers import BertTokenizer,BertModel
from transformers import BertTokenizer, BertModel

MODELNAME='model_hub/chinese-bert-wwm-ext' #ok

tokenizer = BertTokenizer.from_pretrained(MODELNAME)  # 分词词
model = BertModel.from_pretrained(MODELNAME)  # 模型
model.to("cuda")

def get_embedding(text):
    sample = tokenizer(text, max_length=64, truncation=True, padding='max_length', return_tensors='pt')
    sample.to("cuda")
    input_ids = sample['input_ids']
    attention_mask = sample['attention_mask']
    token_type_ids = sample['token_type_ids']
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids).last_hidden_state
    return outputs

def similarity(word1, word2):
    print("word1为{}".format(word1))
    print("word2为{}".format(word2))
    word1_embedding = get_embedding(word1)
    #print("----the shape of word1----", word1.shape)
    word2_embedding = get_embedding(word2)
    #print("----the shape of word2----", word2.shape)

    sim = torch.cosine_similarity(word1_embedding.reshape(1, -1), word2_embedding.reshape(1, -1))
    #print("---the similarity----", sim)
    return sim


def calculate_list():
    test_list1 = []
    test_list2 = []

    df = pd.read_csv("data/sat.csv")
    sat = df['卫星']
    for i in sat:
        test_list1.append(i)
    print(test_list1)


    ad = pd.read_csv("data/app.csv")
    app = ad["app"]
    for i in app:
        test_list2.append(i)
    print(test_list2)
    print("------", len(test_list2))
    similarity_result = {}

    with pd.ExcelWriter("bert_test_result_application.xlsx", mode="a", if_sheet_exists="replace") as writer:

        for i in range(len(test_list2)):
            last_sim_list = []
            sensor_list = []
            app_list = []
            last_result = {}
            for j in range(len(test_list1)):
                print("----------处理第{}个应用任务的第{}个卫星------------".format(str(i), str(j)))
                # result_key = test_list1[j] + "和" + test_list2[i] + "的相似度为"
                last_sim_list.append(similarity(test_list1[j], test_list2[i]).item())
                sensor_list.append(test_list1[j])
                app_list.append(test_list2[i])
                # similarity_result[result_key] = similarity(test_list1[j], test_list2[i])

            last_result = {"sensor": sensor_list, "app": app_list, "sim": last_sim_list}
            df = pd.DataFrame(last_result)
            df.to_excel(writer, index=False, sheet_name=test_list2[i])
        # print("-------------------------------", last_result)
        # with pd.ExcelWriter("bert_test_result_application.xlsx", mode='a',if_sheet_exists='replace') as writer:
        #     df = pd.DataFrame(last_result)
        #     df.to_excel(writer, sheet_name=test_list2[i])

    # from collections import OrderedDict
    # similarity_result = OrderedDict(sorted(similarity_result.items(), key=lambda x: x[1]))
    #
    # for key, value in similarity_result.items():
    #
    #     print(key + str(value.item()))

calculate_list()
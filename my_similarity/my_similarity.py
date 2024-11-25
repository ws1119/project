import csv
import random
from datetime import datetime

import pandas as pd

from TransH.similarity import graph_similarity as gs
print(datetime.now())
from SimCSE import simcse_similarity as ss
print(datetime.now())

def similarity(word1, word2):
    # 获取图谱的相似度和SimCSE的相似度
    graph_similarity, in_graph = gs.double_word_similarity(word1, word2)
    bert_simcse_similarity = ss.word_similarity(word1, word2, 1)
    # print("bert_simcse_similarity=", bert_simcse_similarity)
    # print("graph_similarity=", graph_similarity)
    if in_graph:
        last_similarity = bert_simcse_similarity * 0.3 + graph_similarity * 0.7
    else:
        last_similarity = bert_simcse_similarity * 0.7 + graph_similarity * 0.3
    return last_similarity


print(similarity("高位滑坡应急监测", "饮用水源地悬浮物的空间分布特征获取"))
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
    similarity_result = {}

    with pd.ExcelWriter("data/test_result_application_0.7g_0.3s.xlsx", mode="w") as writer:
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


    # from collections import OrderedDict
    # similarity_result = OrderedDict(sorted(similarity_result.items(), key=lambda x: x[1]))
    #
    # for key, value in similarity_result.items():
    #
    #     print(key + str(value.item()))

calculate_list()
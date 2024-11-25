import pandas as pd

def get_similarity(file_path):
    sim_pd = pd.read_excel(file_path, sheet_name=None)
    # 格式为{“应用任务”：{“卫星”：相似度}}
    kg_simcse_sim = {}
    for key, value in sim_pd.items():
        sim = {}
        for index, row in value.iterrows():
            sim[row["sensor"]] = row["sim"]
        kg_simcse_sim[key] = sim
    return kg_simcse_sim

def get_used_satellite():
    used_satellite_pd = pd.read_csv("data/所用卫星.csv")
    app_list = []
    for i in used_satellite_pd["实体1"].drop_duplicates():
        app_list.append(i)
    # print("----", len(app_list))
    # 格式为{“应用任务”：["卫星一","卫星2"...]}
    used_sate_dic = {i: [] for i in app_list}
    for i, j in zip(used_satellite_pd["实体1"], used_satellite_pd["实体2"]):
        used_sate_dic[i].append(j)
    return used_sate_dic

def get_cr(similarity_result, satellite_result):
    # 每个应用任务对应的平均对比度值
    cr_dict = {i: 0.0 for i in satellite_result.keys()}
    sum_cr = 0
    for app_task, used_sate in satellite_result.items():
        related_sim = 0
        related_count = 0
        unrelated_sim = 0
        unrelated_count = 0
        sate_sim = similarity_result[app_task]
        for sate, sim in sate_sim.items():
            if sate in used_sate:
                related_count += 1
                related_sim += sim
            else:
                unrelated_count += 1
                unrelated_sim += sim
        mean_relate_sim = related_sim/related_count
        print(app_task + "11111", mean_relate_sim)
        mean_unrelated_sim = unrelated_sim / unrelated_count
        print(app_task + "222222", mean_unrelated_sim)
        cr = abs(mean_relate_sim - mean_unrelated_sim)
        sum_cr += cr
        cr_dict[app_task] = cr
    mean_cr = sum_cr / len(cr_dict)
    cr_dict["mean_cr"] = mean_cr
    return cr_dict


satellite_result = get_used_satellite()
# bert
# bert_similarity_result = get_similarity("data/bert_test_result_application.xlsx")
# bert_cr = get_cr(bert_similarity_result, satellite_result)
#
# write_data = {"app": list(bert_cr.keys()), "cr": list(bert_cr.values())}
# df = pd.DataFrame(write_data)
# with pd.ExcelWriter("data/bert_cr.xlsx", mode="w") as writer:
#     df.to_excel(writer, sheet_name="bert_cr", index=True)

# graph
graph_similarity_result = get_similarity("data/test_result_application_0.7g_0.3s.xlsx")
graph_cr = get_cr(graph_similarity_result, satellite_result)
write_data = {"app": list(graph_cr.keys()), "cr": list(graph_cr.values())}
df = pd.DataFrame(write_data)
with pd.ExcelWriter("result/graph_cr_0.7g_0.3s.xlsx", mode="w") as writer:
    df.to_excel(writer, sheet_name="graph_cr", index=True)

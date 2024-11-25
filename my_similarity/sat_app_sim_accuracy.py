import pandas as pd

def get_similarity(file_path):
    sim_df = pd.read_excel(file_path, sheet_name=None)
    sim_result = {}
    for key, value in sim_df.items():
        sim = {}
        for index, row in value.iterrows():
            sim[row["sensor"]] = row["sim"]
        sorted_sim = sorted(sim.items(), key=lambda item: item[1], reverse=True)
        sim = dict(sorted_sim)
        sim_result[key] = sim
    return sim_result

def get_used_satellite():
    used_satellite_pd = pd.read_csv("data/所用卫星.csv")
    # used_satellite_pd = pd.read_csv("data/所用卫星.csv")
    app_list = []
    for i in used_satellite_pd["实体1"].drop_duplicates():
        app_list.append(i)
    # print("----", len(app_list))
    # 格式为{“应用任务”：["卫星一","卫星2"...]}
    used_sate_dic = {i: [] for i in app_list}
    for i, j in zip(used_satellite_pd["实体1"], used_satellite_pd["实体2"]):
        used_sate_dic[i].append(j)
    return used_sate_dic

def get_accuracy(used_sate, sim):
    accuracy_dict = {i: 0.0 for i in used_sate.keys()}
    right_count = 0
    for task, value in used_sate.items():
        right_count = 0
        # 算的得到的应用任务与卫星相似度排序长度为应用任务实际所用的卫星的个数计
        sim_sate_result = list(sim[task].keys())[:len(value)]
        print("task" + task + "111sim_sate_result" + str(sim_sate_result))
        print("task" + task + "222value" +str(value))
        for i in sim_sate_result:
            if i in value:
                right_count += 1
        if right_count == 0:
            accuracy = 0
        else:
            accuracy = right_count / len(value)
        accuracy_dict[task] = accuracy
    mean_acc = sum(accuracy_dict.values()) / len(accuracy_dict)
    accuracy_dict["mean_acc"] = mean_acc

    print("-----accuracy-----", accuracy_dict)
    print(mean_acc)
    return  accuracy_dict


used_sate = get_used_satellite()

sim_result = get_similarity("data/test_result_application_0.7g_0.3s.xlsx")
# sim_result = get_similarity("data/test_result_application1.xlsx")
graph_acc = get_accuracy(used_sate, sim_result)


write_data = {"app": list(graph_acc.keys()), "cr": list(graph_acc.values())}
df = pd.DataFrame(write_data)
with pd.ExcelWriter("result/graph_acc_0.7g_0.3s.xlsx", mode="w") as writer:
    df.to_excel(writer, sheet_name="graph_acc", index=True)
# print(sim_result)
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from my_similarity.SimCSE import simcse_similarity



from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertConfig, BertTokenizer
import torch.nn.functional as F


model_path = "SimCSE/model_set/chinese-bert-wwm-ext"
save_path = "SimCSE/model_saved/best_model_outputway-cls_batch-128_lr-5e-05.pth"
output_way = 'cls'
tokenizer = BertTokenizer.from_pretrained(model_path)
Config = BertConfig.from_pretrained(model_path)
maxlen = 64
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class NeuralNetwork(nn.Module):
    def __init__(self,model_path,output_way):
        super(NeuralNetwork, self).__init__()
        self.bert = BertModel.from_pretrained(model_path,config=Config)
        self.output_way = output_way
    def forward(self, input_ids, attention_mask, token_type_ids):
        # print("---inputids---", input_ids.shape)
        # print("---attention_mask---", attention_mask.shape)
        # print("---token_type_ids---", token_type_ids.shape)
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.output_way == 'cls':
            output = x1.last_hidden_state[:,0]
        elif self.output_way == 'pooler':
            output = x1.pooler_output
        return output



class InferenceDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法"""

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return tokenizer(text, max_length=maxlen, truncation=True, padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        da = self.data[index]
        return self.text_2_id([da])

MODEL = NeuralNetwork(model_path, output_way)
MODEL.load_state_dict(torch.load(save_path))


def get_vector_simcse(sentence, model_id):
    """
    预测simcse的语义向量。
    """
    model = None
    if model_id == 1:
        model = MODEL
    data_feed = list(DataLoader(InferenceDataset([sentence]), batch_size=1))[0]
    model.eval()
    with torch.no_grad():
        input_ids = data_feed.get('input_ids').squeeze(1)
        attention_mask = data_feed.get('attention_mask').squeeze(1)
        token_type_ids = data_feed.get('token_type_ids').squeeze(1)
        output_vector = model(input_ids, attention_mask, token_type_ids)
    return output_vector#[0]


def get_simcse_word_embedding(word, model_id=1):
    return get_vector_simcse(word, model_id)


def simcse_word_similarity(word1, word2, model_id):
    word1_embedding = get_vector_simcse(word1, model_id)
    word2_embedding = get_vector_simcse(word2, model_id)
    word1_tensor = torch.Tensor(word1_embedding)
    word2_tensor = torch.Tensor(word2_embedding)

    word1_norm = F.normalize(word1_tensor, p=2)
    word2_norm = F.normalize(word2_tensor, p=2)


    cos_sim = torch.cosine_similarity(word2_norm, word1_norm)
    #print("====", cos_sim)
    return cos_sim


# 实体及其相对应的embedding字典
entity_embedding_dict = {}
# 图谱中的实体列表
entity_list = []
# 实体的embedding列表
embedding_list = []

with open("TransH/data/entity2id.txt", "r", encoding="utf-8") as fp:
    s = fp.readlines()
    for entity2id in s:
        entity = entity2id[0:entity2id.find("\t")]
        entity_list.append(entity)

#with open("TransH/result_torch/epoch1200_TransH_pytorch_entity_200dim_batch1200", "r") as fp:
with open("TransH/result_torch/epoch300_TransH_pytorch_entity_768dim_batch50", "r") as fp:
# with open("TransH/result_torch/SelectE_result_all.txt", "r") as fp:
    s = fp.readlines()
    for embedding in s:
        tmp = []
        # 将存储的Embedding去除[],并按照，进行分割，得到每一个值
        for j in embedding[embedding.find("[") + 1:-2].split(","):
            tmp.append(j)
        embedding_list.append(tmp)

for i in range(len(entity_list)):
    entity_embedding_dict[entity_list[i]] = embedding_list[i]

# 获取图谱中节点的embedding
def get_graph_word_embedding(find_word):
    find_word_embedding = entity_embedding_dict[find_word]
    # 将元素转化为float
    find_word_embedding = list(map(float, find_word_embedding))
    return find_word_embedding

def longest_common_subsequence(text1, text2):
    m = len(text1)
    n = len(text2)

    # 创建动态规划表，dp[i][j]表示text1的前i个字符和text2的前j个字符的最长公共子序列长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 填充动态规划表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 回溯找到最长公共子序列
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # 最长公共子序列为lcs的逆序
    lcs.reverse()

    # 返回最长公共子序列及其长度
    return "".join(lcs), dp[m][n]

# 通过最长公共序列获取没在图谱中的词语的向量
def graph_weighted_average_embedding(input):
    weight_is_empty = False
    word_not_in_graph_embedding_list = []
    # 加权平均的权重列表
    weights = []
    for text1 in entity_list:
        lcs, lenght = longest_common_subsequence(text1, input)
        lenght1 = longest_common_subsequence(text1, input)[1]
        #print("{}与{}的最长公共子序列为{}，长度为{}".format(input, text1, lcs, lenght1))
        if lenght1 >= 2:
            #lenght1 = (lenght1 * 2) / (len(input) + len(text1))
            # 直接以最长公共子序列的长度作为权重
            weights.append(lenght1)
            #print("{}与{}的最长公共子序列为{}，长度为{}".format(input, text1, lcs, lenght1))
            word_not_in_graph_embedding_list.append(np.array(entity_embedding_dict[text1]))
    # 权重列表为空直接则直接返回
    if not weights:
        print(input + "权重没有")
        return [], False
    else:
        embedding_matrix = np.array(word_not_in_graph_embedding_list, dtype=np.float64)
        # 获取词嵌入向量
        word_embeddings = [embedding_matrix[word_index] for word_index in range(len(embedding_matrix))]
        # 使用加权平均法计算短语的词嵌入
        weighted_average = np.average(np.array(word_embeddings), axis=0, weights=weights)
        return weighted_average, True


# 计算某一个词的相似度排序
def graph_word_similarity(word):
    print("图谱中的节点为：", entity_list)
    sim_dict = {}
    # 如果该词语在图谱中存在，直接计算该词语的相似度，排序
    if word in entity_list:
        print("输入的词语在图谱中")
        # 获取输入的word的embedding
        word_embedding = torch.Tensor(get_graph_word_embedding(word)).reshape(1, -1)
        normalized_word_embedding = F.normalize(word_embedding, p=2)
        for other_word in entity_list:
            other_word_embedding = torch.Tensor(get_graph_word_embedding(other_word)).reshape(1, -1)
            normalized_other_word_embedding = F.normalize(other_word_embedding, p=2)
            # 余弦相似度
            cosine_sim = torch.sum(normalized_word_embedding * normalized_other_word_embedding)
            # cosine_sim = torch.nn.functional.cosine_similarity(normalized_other_word_embedding, normalized_word_embedding)
            sim_dict["{}--{}".format(word, other_word)] = cosine_sim.item()
    else: # 该词语在图谱中不存在，通过最长公共子序列获取词向量计算相似度
        print("输入的词语不在图谱中")
        # 获取输入的word的embedding
        # print("weighted_average_embedding(word)",weighted_average_embedding(word))
        word_embedding = torch.Tensor(graph_weighted_average_embedding(word)).reshape(1, -1)
        normalized_word_embedding = F.normalize(word_embedding, p=2)
        for other_word in entity_list:
            other_word_embedding = torch.Tensor(graph_weighted_average_embedding(other_word)).reshape(1, -1)
            normalized_other_word_embedding = F.normalize(other_word_embedding, p=2)
            # 余弦相似度
            cosine_sim = torch.sum(normalized_word_embedding * normalized_other_word_embedding)
            sim_dict["{}--{}".format(word, other_word)] = cosine_sim.item()
        # 按照相似度排序
    sim_dict = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
    return sim_dict

def graph_double_word_similarity(word1, word2):
    in_graph = False
    if word1 in entity_list and word2 in entity_list:
        print("{}和{}都在图谱中".format(word1, word2))
        word1_embedding = torch.Tensor(get_graph_word_embedding(word1)).reshape(1, -1)
        word2_embedding = torch.Tensor(get_graph_word_embedding(word2)).reshape(1, -1)
        in_graph = True
    elif word1 in entity_list and word2 not in entity_list:
        print(f"{word1}在图谱中，{word2}不在图谱中")
        # word2不在图谱中，且与图谱中的几点不存在LCS，则使用simcse计算相似度
        if not graph_weighted_average_embedding(word2)[1]:
            word1_embedding = torch.Tensor(get_simcse_word_embedding(word1)).reshape(1, -1)
            word2_embedding = torch.Tensor(get_simcse_word_embedding(word2)).reshape(1, -1)
        else:
            word1_embedding = torch.Tensor(get_graph_word_embedding(word1)).reshape(1, -1)
            word2_embedding = torch.Tensor(graph_weighted_average_embedding(word2)[0]).reshape(1, -1)
    elif word1 not in entity_list and word2 in entity_list:
        print(f"{word2}在图谱中，{word1}不在图谱中")
        if not graph_weighted_average_embedding(word1)[1]:
            word1_embedding = torch.Tensor(get_simcse_word_embedding(word1)).reshape(1, -1)
            word2_embedding = torch.Tensor(get_simcse_word_embedding(word2)).reshape(1, -1)
        else:
            word1_embedding = torch.Tensor(get_graph_word_embedding(word2)).reshape(1, -1)
            word2_embedding = torch.Tensor(graph_weighted_average_embedding(word1)[0]).reshape(1, -1)
    else:
        print("{}和{}都不在图谱中".format(word1, word2))
        if graph_weighted_average_embedding(word1)[1] or graph_weighted_average_embedding(word2)[1]:
            word1_embedding = torch.Tensor(get_simcse_word_embedding(word1)).reshape(1, -1)
            word2_embedding = torch.Tensor(get_simcse_word_embedding(word2)).reshape(1, -1)
        else:
            word1_embedding = torch.Tensor(graph_weighted_average_embedding(word1)[0]).reshape(1, -1)
            word2_embedding = torch.Tensor(graph_weighted_average_embedding(word2)[0]).reshape(1, -1)
    normalized_word1_embedding = F.normalize(word1_embedding, p=2)
    normalized_word2_embedding = F.normalize(word2_embedding, p=2)
    cos_sim = torch.sum(normalized_word1_embedding * normalized_word2_embedding)

    return cos_sim, in_graph


def similarity(word1, word2):
    # 获取图谱的相似度和SimCSE的相似度
    graph_similarity, in_graph = graph_double_word_similarity(word1, word2)
    bert_simcse_similarity = simcse_word_similarity(word1,word2, 1)
    #lstm_simcse_similarity = ss.word_similarity("城市水体提取","黑臭水体识别", 2)
    print("----bert_simcse_similarity=", bert_simcse_similarity)
    # print("lstm_simcse_similarity=", lstm_simcse_similarity)
    print("----graph_similarity=", graph_similarity)
    if in_graph:
        last_similarity = bert_simcse_similarity * 0.5 + graph_similarity * 0.5
    else:
        last_similarity = bert_simcse_similarity * 0.5 + graph_similarity * 0.5
    #if in_graph:
     #   last_similarity = bert_simcse_similarity * 0.1 + graph_similarity * 0.9

    print("----last_similarity", last_similarity)
    return last_similarity


def get_test_list():
    list1 = []
    list2 = []
    with open("data/test2.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            list1.append(str(row[0]))
            list2.append(str(row[1]))
    return list1, list2


test_list1, test_list2 = get_test_list()
print(test_list1)
print(test_list2)

similarity_result = {}
for i in range(len(test_list1)):
    result_key = test_list1[i] + "和" + test_list2[i] + "的相似度为"
    with open("result/simcse_similarity_result_with_TransH_new.txt", "a") as f:
        f.write(str(similarity(test_list1[i], test_list2[i]).item()) + "\n")
    # similarity_result[result_key] = similarity(test_list1[i], test_list2[i])

# from collections import OrderedDict
# similarity_result = OrderedDict(sorted(similarity_result.items(), key=lambda x: x[1]))
# # print(sorted_d)
#
# for key, value in similarity_result.items():
#     print(key + str(value.item()))


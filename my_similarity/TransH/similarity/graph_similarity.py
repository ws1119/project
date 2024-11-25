import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from my_similarity.SimCSE import simcse_similarity


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

with open("TransH/result_torch/epoch300_TransH_pytorch_entity_768dim_batch50", "r") as fp:

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
def get_word_embedding(find_word):
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
def weighted_average_embedding(input):
    # weight_is_empty = False
    word_not_in_graph_embedding_list = []
    # 加权平均的权重列表
    weights = []
    for text1 in entity_list:
        lcs, lenght = longest_common_subsequence(text1, input)
        lenght1 = longest_common_subsequence(text1, input)[1]
        #print("{}与{}的最长公共子序列为{}，长度为{}".format(input, text1, lcs, lenght1))
        if lenght1 >= 1:
            #lenght1 = (lenght1 * 2) / (len(input) + len(text1))
            # 直接以最长公共子序列的长度作为权重
            weights.append(lenght1)
            #print("{}与{}的最长公共子序列为{}，长度为{}".format(input, text1, lcs, lenght1))
            word_not_in_graph_embedding_list.append(np.array(entity_embedding_dict[text1]))
    # # 权重列表为空直接则直接返回
    # if not weights:
    #     return [], False
    else:
        embedding_matrix = np.array(word_not_in_graph_embedding_list, dtype=np.float64)
        # 获取词嵌入向量
        word_embeddings = [embedding_matrix[word_index] for word_index in range(len(embedding_matrix))]
        # 使用加权平均法计算短语的词嵌入
        weighted_average = np.average(np.array(word_embeddings), axis=0, weights=weights)

        return weighted_average


# 计算某一个词的相似度排序
def word_similarity(word):
    print("图谱中的节点为：", entity_list)
    sim_dict = {}
    # 如果该词语在图谱中存在，直接计算该词语的相似度，排序
    if word in entity_list:
        print("输入的词语在图谱中")
        # 获取输入的word的embedding
        word_embedding = torch.Tensor(get_word_embedding(word)).reshape(1, -1)
        normalized_word_embedding = F.normalize(word_embedding, p=2)
        for other_word in entity_list:
            other_word_embedding = torch.Tensor(get_word_embedding(other_word)).reshape(1, -1)
            normalized_other_word_embedding = F.normalize(other_word_embedding, p=2)
            # 余弦相似度
            cosine_sim = torch.sum(normalized_word_embedding * normalized_other_word_embedding)
            # cosine_sim = torch.nn.functional.cosine_similarity(normalized_other_word_embedding, normalized_word_embedding)
            sim_dict["{}--{}".format(word, other_word)] = cosine_sim.item()
    else: # 该词语在图谱中不存在，通过最长公共子序列获取词向量计算相似度
        print("输入的词语不在图谱中")
        # 获取输入的word的embedding
        # print("weighted_average_embedding(word)",weighted_average_embedding(word))
        word_embedding = torch.Tensor(weighted_average_embedding(word)).reshape(1, -1)
        normalized_word_embedding = F.normalize(word_embedding, p=2)
        for other_word in entity_list:
            other_word_embedding = torch.Tensor(get_word_embedding(other_word)).reshape(1, -1)
            normalized_other_word_embedding = F.normalize(other_word_embedding, p=2)
            # 余弦相似度
            cosine_sim = torch.sum(normalized_word_embedding * normalized_other_word_embedding)
            # cosine_sim = torch.nn.functional.cosine_similarity(normalized_other_word_embedding, normalized_word_embedding)
            sim_dict["{}--{}".format(word, other_word)] = cosine_sim.item()
        # 按照相似度排序
    sim_dict = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
    return sim_dict

def double_word_similarity(word1, word2):
    in_graph = False
    if word1 in entity_list and word2 in entity_list:
        # print("{}和{}都在图谱中".format(word1, word2))
        word1_embedding = torch.Tensor(get_word_embedding(word1)).reshape(1, -1)
        word2_embedding = torch.Tensor(get_word_embedding(word2)).reshape(1, -1)
        in_graph = True
    elif word1 in entity_list and word2 not in entity_list:
        # print(f"{word1}在图谱中，{word2}不在图谱中")
        word1_embedding = torch.Tensor(get_word_embedding(word1)).reshape(1, -1)
        word2_embedding = torch.Tensor(weighted_average_embedding(word2)).reshape(1, -1)
    elif word1 not in entity_list and word2 in entity_list:
        # print(f"{word2}在图谱中，{word1}不在图谱中")
        word1_embedding = torch.Tensor(weighted_average_embedding(word1)).reshape(1, -1)

        word2_embedding = torch.Tensor(get_word_embedding(word2)).reshape(1, -1)
    else:
        # print("{}和{}都不在图谱中".format(word1, word2))
        word1_embedding = torch.Tensor(weighted_average_embedding(word1)).reshape(1, -1)
        word2_embedding = torch.Tensor(weighted_average_embedding(word2)).reshape(1, -1)
    normalized_word1_embedding = F.normalize(word1_embedding, p=2)
    normalized_word2_embedding = F.normalize(word2_embedding, p=2)
    cos_sim = torch.sum(normalized_word1_embedding * normalized_word2_embedding)
    return cos_sim, in_graph


# def double_word_similarity(word1, word2):
#     word1_embedding = torch.Tensor(weighted_average_embedding(word1)).reshape(1, -1)
#     word2_embedding = torch.Tensor(weighted_average_embedding(word2)).reshape(1, -1)
#     normalized_word1_embedding = F.normalize(word1_embedding, p=2)
#     normalized_word2_embedding = F.normalize(word2_embedding, p=2)
#     cos_sim = torch.sum(normalized_word1_embedding * normalized_word2_embedding)
#     return cos_sim



# # 获取输入的词语与图谱中所有节点的相似度
# result = word_similarity("城市黑臭水体识别")
# for i in range(10):
#     print(result[i])

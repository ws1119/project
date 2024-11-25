import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 实体及其相对应的embedding字典
entity_embedding_dict = {}
# 图谱中的实体列表
entity_list = []
# 实体的embedding列表
embedding_list = []

with open("../data/entity2id.txt", "r") as fp:
    s = fp.readlines()
    for entity2id in s:
        entity = entity2id[0:entity2id.find("\t")]
        entity_list.append(entity)
print(entity_list)

with open("../result_torch/epoch1200_TransH_pytorch_entity_200dim_batch1200", "r") as fp:
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

# 计算图谱中存在的词的相似度排序
def calculate_similarity(word):
    # 存放所有节点相似度
    sim_dict = {}
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

    # 按照相似度排序
    sim_dict = sorted(sim_dict.items(), key=lambda x:x[1], reverse=True)
    return sim_dict
    print(sim_dict)
    # normalized_tensor_1 = F.normalize(input1, p=2)
    # print(normalized_tensor_1)
    # cosine_sim = torch.sum(normalized_tensor_1 * normalized_tensor_2)
    # print("pytorch_sim", cosine_sim)

# 计算两个词的相似度
def two_word_similarity(word1, word2):
    word1_embedding = torch.Tensor(get_word_embedding(word1)).reshape(1, -1)
    word2_embedding = torch.Tensor(get_word_embedding(word2)).reshape(1, -1)

    word1_embedding_norm = F.normalize(word1_embedding, 2)
    word2_embedding_norm = F.normalize(word2_embedding, 2)

    cosine_sim = torch.sum(word1_embedding_norm * word2_embedding_norm)

    return cosine_sim

# 最长公共子序列
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
    word_not_in_graph_embedding_list = []
    # 加权平均的权重列表
    weights = []
    for text1 in entity_list:
        lcs, lenght = longest_common_subsequence(text1, input)
        lenght1 = longest_common_subsequence(text1, input)[1]
        if lenght1 >= 3:
            lenght1 = (lenght1 * 2) / (len(input) + len(text1))
            weights.append(lenght1)
            print("{}与{}的最长公共子序列为{}，长度为{}".format(input, text1, lcs, lenght1))
            word_not_in_graph_embedding_list.append(np.array(entity_embedding_dict[text1]))
    embedding_matrix = np.array(word_not_in_graph_embedding_list, dtype=np.float64)
    # 获取词嵌入向量
    word_embeddings = [embedding_matrix[word_index] for word_index in range(len(embedding_matrix))]
    # 使用加权平均法计算短语的词嵌入
    weighted_average = np.average(np.array(word_embeddings), axis=0, weights=weights)
    return weighted_average


# sim_result = calculate_similarity("水体识别")
#
# for i in range(50):
#     print("+++",sim_result[i])
#
# print("=====", two_word_similarity("水体识别", "城市黑臭水体识别"))
#
# weighted_average = weighted_average_embedding("目标检测")
# # print("加权平均法计算的短语的词嵌入:", weighted_average)
#
# output_embedding = torch.Tensor(weighted_average).reshape(1, -1)
# output_embedding_norm = F.normalize(output_embedding, 2)
# test_em = torch.Tensor(get_word_embedding("船舶目标检测")).reshape(1, -1)
# test_em_norm = F.normalize(test_em, 2)
# print("sim=", torch.sum(output_embedding_norm * test_em_norm))




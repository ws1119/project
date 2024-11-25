from datetime import datetime
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertConfig, BertTokenizer
import torch.nn.functional as F


model_path = "SimCSE/model_set/chinese-bert-wwm-ext"
save_path = "SimCSE/model_saved/best_model_outputway-cls_batch-128_lr-5e-05_dropout-0.1-3"
output_way = 'cls'
tokenizer = BertTokenizer.from_pretrained(model_path)
Config = BertConfig.from_pretrained(model_path)
maxlen = 64
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class NeuralNetwork(nn.Module):
    def __init__(self,model_path,output_way):
        super(NeuralNetwork, self).__init__()
        self.bert = BertModel.from_pretrained(model_path,config=Config).to(device)
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
        return tokenizer(text, max_length=maxlen, truncation=True, padding='max_length', return_tensors='pt').to(device)

    def __getitem__(self, index: int):
        da = self.data[index]
        return self.text_2_id([da])

MODEL = NeuralNetwork(model_path, output_way)
MODEL.load_state_dict(torch.load(save_path))
model = MODEL
model.to(device)

def get_vector_simcse(sentence, model_id):
    """
    预测simcse的语义向量。
    """

    data_feed = list(DataLoader(InferenceDataset([sentence]), batch_size=1))[0]
    model.eval()
    with torch.no_grad():
        input_ids = data_feed.get('input_ids').squeeze(1)
        attention_mask = data_feed.get('attention_mask').squeeze(1)
        token_type_ids = data_feed.get('token_type_ids').squeeze(1)
        output_vector = model(input_ids, attention_mask, token_type_ids)
    return output_vector#[0]



def get_word_embedding(word, model_id=1):
    return get_vector_simcse(word, model_id)

def word_similarity(word1, word2, model_id):
    word1_embedding = get_vector_simcse(word1, model_id)
    word2_embedding = get_vector_simcse(word2, model_id)
    word1_tensor = torch.Tensor(word1_embedding)
    word2_tensor = torch.Tensor(word2_embedding)

    word1_norm = F.normalize(word1_tensor, p=2)
    word2_norm = F.normalize(word2_tensor, p=2)


    cos_sim = torch.cosine_similarity(word2_norm, word1_norm)
    #print("====", cos_sim)
    return cos_sim

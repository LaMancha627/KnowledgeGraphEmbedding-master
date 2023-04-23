import os

from torch.nn import TransformerEncoder

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F



# 创建一个LSTM模型，它接收路径的嵌入表示并输出一个向量表示
class PathLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(PathLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return h.squeeze()


'''
# 初始化模型和损失函数
embedding_size = 32
hidden_size = 16
model = PathLSTM(len(vocab) + 1, embedding_size, hidden_size)
'''


# 创建数据集和数据加载器，并使用批处理将所有路径分组
class PathDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]


# dataset = PathDataset(padded_sequences)
# dataloader = DataLoader(dataset, batch_size=32)

'''
# 计算所有路径的向量表示并打印
with torch.no_grad():
    for paths in dataloader:
        vectors = model(paths)
        # print(vectors)
'''
import torch
import torch.nn as nn
import math


class PathTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(PathTransformer, self).__init__()

        # 位置编码向量维度
        self.d_model = d_model
        # 注意力头数
        self.nhead = nhead
        # Transformer 层数
        self.num_layers = num_layers

        # 位置编码层
        self.pos_encoder = PositionalEncoding(d_model)
        # 输入嵌入层
        self.input_embed = nn.Linear(d_model, d_model)
        # Transformer 层
        self.transformer_layers = nn.TransformerEncoderLayer(d_model, nhead)
        # Transformer 模型
        self.transformer = nn.TransformerEncoder(self.transformer_layers, num_layers)
        # 输出层
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, path):
        # path shape: (seq_len, batch_size, d_model)

        # 位置编码
        path = self.pos_encoder(path)

        # 输入嵌入
        path = self.input_embed(path)

        # Transformer 层
        path = self.transformer(path)

        # 输出层
        path = self.output_layer(path)

        # 返回输出
        return path.squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # 注册位置编码矩阵为可学习参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


import torch

# from transformers import TransformerEncoder
'''
# 定义模型参数
# input_dim = 256   # 输入向量的维度
input_dim = 16  # 输入向量的维度
# 与lstm的hidden_size大小相同？
# hidden_dim = 512  # Transformer 编码器中每个头的维度
hidden_dim = 8  # Transformer 编码器中每个头的维度
num_layers = 4  # Transformer 编码器的层数
max_len = 100  # 输入序列的最大长度

# 定义模型
model1 = PathTransformer(input_dim, hidden_dim, num_layers)
'''
'''# 生成输入数据
batch_size = 1
seq_len = 50
input_data = torch.randn(batch_size, seq_len, input_dim)
'''
'''
# 进行编码
path1 = []
with torch.no_grad():
    for paths in dataloader:
        vectors = model(paths)
        output = model1(vectors)
        path_embeds_norm = F.normalize(output, p=2, dim=1)
        # print(path_embeds_norm.size())
        # path1 = torch.cat([path1, path_embeds_norm], dim=1).
        path1.append(path_embeds_norm)
        # print(output)
'''
'''
        wn18前1/3
        print(path1[1860].size())
        #path1[1860] = torch.nn.functional.pad(path1[1860], (0, 2), 'constant', 0)
        #path1[1860] = torch.nn.functional.pad(path1[1860], (1, 2), 'constant', 0)
        y = torch.zeros(32, 32)  # 创建一个形状为 [32, 32] 的空张量
        # 将 x 复制到 y 的左上角
        y[:30, :30] = path1[1860]
        # 现在 y 的形状是 [32, 32]，但是最后两行和最后两列都是空的，需要裁剪掉
        #y = y[:30, :30]
        print(y.shape)  # 输出 torch.Size([32, 32])
        #print(path1[1860].size())
        path1[1860] = y
        print(path1[1860].size())
       
    #print(path1[1413].size())
    # path1[1860] = torch.nn.functional.pad(path1[1860], (0, 2), 'constant', 0)
    # path1[1860] = torch.nn.functional.pad(path1[1860], (1, 2), 'constant', 0)
    y = torch.zeros(32, 32)  # 创建一个形状为 [32, 32] 的空张量
    # 将 x 复制到 y 的左上角
    y[:path1[1413].size()[0], :path1[1413].size()[1]] = path1[1413]
    # 现在 y 的形状是 [32, 32]，但是最后两行和最后两列都是空的，需要裁剪掉
    # y = y[:30, :30]
    #print(y.shape)  # 输出 torch.Size([32, 32])
    # print(path1[1860].size())
    path1[1413] = y
    #print(path1[1413].size())
'''
'''unified_shape = [32, 32]

    # 找出形状不同的张量，并统一形状
    for tensor in path1:
        if unified_shape is None or tensor.shape != unified_shape:
            unified_shape = tensor.shape
            # 修改形状不一致的张量
            tensor = tensor.view(unified_shape)
'''

# concatenated_tensor = torch.cat([x for x in path1], dim=1)

# torch.save(concatenated_tensor, 'path_tensor.pt')
# print(concatenated_tensor)
# print(concatenated_tensor.size())
# 输出编码后的路径信息表示
# print(output)

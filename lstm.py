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

print(os.getcwd())

with open('data/wn18/entities.dict') as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)

with open('data/wn18/relations.dict') as fin:
    relation2id = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def get_degree(triples, degree=0):
    '''
    获取实体的度
    '''
    '''for head, relation, tail in triples:
        if head in triples:
            degree += 1
            print(head + "度为:   " + str(degree))
        degree = 0'''
    degree_dict = {}
    a = []
    for head, relation, tail in triples:
        # print(type(triples))
        for i in triples:
            '''#print(i[0])
            if head == i[0] or head == i[2]:
                degree += 1
        degree_dict[head]=degree
        print(str(head)+'degree:   '+str(degree))
        degree = 0'''
            a[i[0]] += 1
            a[i[2]] += 1
    sorted_dict = sorted(degree_dict.items(), key=lambda kv: kv[1])
    #print(sorted_dict)
    nentity = len(entity2id)
    nrelation = len(relation2id)
    return degree


def get_network(triples):
    G = nx.Graph()
    for t in triples:
        G.add_node(t[0])
        G.add_node(t[2])
        G.add_edge(t[0], t[2])  # 结点，结点
    # print("所有节点的度:", G.degree)  # 返回所有节点的度
    # print("所有节点的度分布序列:", nx.degree_histogram(G))  # 返回图中所有节点的度分布序列（从1至最大度的出现次数）
    # 无向图度分布曲线
    degree = nx.degree_histogram(G)
    remove = [node for node, degree in dict(G.degree).items() if degree < 30]
    G.remove_nodes_from(remove)
    #print(type(remove))
    degrees = [(node, val) for (node, val) in G.degree]
    sorted_degree = sorted(degrees, key=lambda x: x[1], reverse=True)
    # print(type(sorted_degree)) list
    #print("排序后：" + str(sorted_degree))
    # 选择前三分之一的元素
    index1 = int(len(sorted_degree) / 3)
    index2 = int(2 * len(sorted_degree) / 3)
    # remain = sorted_degree[0:index1]
    # remain =sorted_degree[index1:index2]
    remain =sorted_degree[index2:]
    # print("保留的节点为：  "+str(remain))
    # remain = [node for node, degree in dict(G.degree).items() if degree > 30]
    # print(remain)
    # print(remain[0])
    remain_triples = []
    i = 0
    # 获取保留节点对应的三元组
    # print("三元组"+str(triples[1][0]))
    for j in range(len(triples)):
        for i in range(len(remain)):
            if triples[j][0] == remain[i][0] or triples[j][0] == remain[i][0]:
                # print(triples[j])
                remain_triples.append(triples[j])
    # print("剩下的三元组有:" + str(remain_triples))
    # print("所有节点的度:", G.degree)  # 返回所有节点的度
    '''  x = range(len(degree))  # 生成X轴序列，从1到最大度
    y = [z / float(sum(degree)) for z in degree]  # 将频次转化为频率，利用列表内涵
    plt.loglog(x, y, color="blue", linewidth=2)  # 在双对坐标轴上绘制度分布曲线
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.title('数据集节点度分布图')
    plt.xlabel("degree")
    plt.ylabel("frequency")
    plt.show()  # 显示图表'''

    return remain_triples


'''def counter(triple):
    return Counter(triple)'''


def get_path(triples):
    G = nx.Graph()
    for t in triples:
        G.add_node(t[0])
        G.add_node(t[2])
        G.add_edge(t[0], t[2])  # 结点，结点
    # 定义node2vec算法参数
    p = 1
    q = 1
    dimensions = 8
    num_walks = 10
    walk_length = 80

    # 使用node2vec算法生成随机游走序列
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q)
    walks = node2vec.walks

    '''# 输出随机游走序列
    for walk in walks:
        print(walk)'''

    return walks


import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import nn

triple = read_triple('data/wn18/train.txt', entity2id, relation2id)
triples = get_network(triple)

# 假设我们有一些路径数据存储在一个列表中，每个路径都是由节点ID组成的列表
path_data = get_path(triples)

# 创建词典，将每个节点ID映射到一个唯一的整数ID
vocab = {node_id: i + 1 for i, node_id in enumerate(set([node for path in path_data for node in path]))}

# 将每个路径转换为一个序列，其中每个节点ID都被其整数ID替换
path_sequences = [[vocab[node] for node in path] for path in path_data]

# 对所有序列使用填充将它们扩展到相同的长度
max_len = max([len(seq) for seq in path_sequences])
padded_sequences = pad_sequence([torch.LongTensor(seq) for seq in path_sequences], batch_first=True, padding_value=0)


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


# 初始化模型和损失函数
embedding_size = 32
hidden_size = 16
model = PathLSTM(len(vocab) + 1, embedding_size, hidden_size)


# 创建数据集和数据加载器，并使用批处理将所有路径分组
class PathDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]


dataset = PathDataset(padded_sequences)
dataloader = DataLoader(dataset, batch_size=32)

# 计算所有路径的向量表示并打印
with torch.no_grad():
    for paths in dataloader:
        vectors = model(paths)
        # print(vectors)

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
    def __init__(self, d_model, max_len=5000):
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

'''# 生成输入数据
batch_size = 1
seq_len = 50
input_data = torch.randn(batch_size, seq_len, input_dim)
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
        '''
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
    '''unified_shape = [32, 32]

    # 找出形状不同的张量，并统一形状
    for tensor in path1:
        if unified_shape is None or tensor.shape != unified_shape:
            unified_shape = tensor.shape
            # 修改形状不一致的张量
            tensor = tensor.view(unified_shape)'''

    concatenated_tensor = torch.cat([x for x in path1], dim=1)

torch.save(concatenated_tensor, 'path_tensor.pt')
    #print(concatenated_tensor)
    #print(concatenated_tensor.size())
# 输出编码后的路径信息表示
# print(output)

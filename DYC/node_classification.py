import logging
import random
import sys

import gensim
import pandas as pd
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

def random_walk(G, v, l):
    """
    游走采样：具体的游走规则是：每次从当前节点的邻居节点中选择一个节点作为下一个节点，选择概率与邻居节点的入度成反比；如果有多个候选节点，则选择度最大的那个作为下一个节点。如果当前节点没有邻居节点或没有候选节点，则结束游走。
    G: 图    v: 游走的根节点    l: 游走的序列的长度
    """
    walk_seq = [v]
    while len(walk_seq) < l:
        node = walk_seq[-1]
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 0:
            candidates = []
            for s in neighbors:
                if s not in walk_seq and G.degree(s) > 0:  # 检查入度是否大于0
                    m = 1 / G.degree(s)
                    if np.random.uniform(0, 1) < m:
                        candidates.append(s)

                        # 如果candidates为空，则终止游走或采取其他策略
            if not candidates:
                break  # 或者可以选择一个已经访问过的节点，或者重新开始游走等。

            # 从candidates中选择一个出度最大的节点
            max_out_degree = -1
            max_out_degree_node = None
            for candidate in candidates:
                degree = G.degree(candidate)
                if degree > max_out_degree:
                    max_out_degree = degree
                    max_out_degree_node = candidate

            walk_seq.append(max_out_degree_node)
        else:
            # 没有邻居节点时，可以选择终止游走或采取其他策略
            break  # 或者可以选择一个已经访问过的节点等。

    return walk_seq


def deep_walk(G, t, d, w, epochs=5):
    """
    深度随机游走。
    G: 图
    t: 每个节点随机游走的次数
    l: 随机游走最大的序列长度
    d: 节点嵌入向量的维度
    w: skipgram的窗口大小
    epochs: skipgram: 训练轮数
    """
    walk_seq_list = []
    nodes = list(G.nodes)
    for walk_length in walk_lengths:
        logger.info(f'当前最大游走序列长度为{walk_length}。')
        logger.info(f'进行采样游走序列。')
        for i in range(t):
            random.shuffle(nodes)
            for node in nodes:
                walk_seq = random_walk(G, node, walk_length)
                walk_seq_list.append(walk_seq)

    logger.info('游走序列采样完成。开始训练skipgram模型...')
    model = gensim.models.Word2Vec(sentences=walk_seq_list, vector_size=d, window=w, max_vocab_size=len(G), min_count=0,
                                   epochs=epochs, sg=1, hs=0)
    return model

def read_node_labels(file_path):
    node_labels = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:  # 确保每行都至少有两个部分
                node = parts[0]
                label = parts[1]
                node_labels[node] = label
            else:
                print("Warning: Line does not contain node label pair:", line)
    return node_labels

# 然后使用已训练的分类器模型预测节点标签，并使用micro-F1和macro-F1 指标来评估算法性能

def main():
    # Generate node embeddings using DeepWalk
    embedding_model = deep_walk(G, t, d, w)

    # Get node embeddings
    embeddings = {str(node): embedding_model.wv[str(node)] for node in G.nodes()}

    # Generate node labels
    node_labels = read_node_labels(file_path)

    # Prepare vectors and labels
    X = np.array([embeddings[str(node)] for node in G.nodes()])
    y = np.array([node_labels[str(node)] for node in G.nodes()])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9,random_state=42)

    #训练一个逻辑回归模型来构建一个多类别分类器，配置使用liblinear 求解器，并进行10000 次最大迭代来寻找最佳模型参数。
    model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=10000, multi_class='ovr')
    model.fit(X_train, y_train)

    # 然后使用已训练的分类器模型预测节点标签，并使用micro-F1和macro-F1 指标来评估算法性能
    y_pred = model.predict(X_test)

    # Calculate micro-F1 and macro-F1 scores
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print("Micro-F1 Score:", micro_f1)
    print("Macro-F1 Score:", macro_f1)
    return  micro_f1,macro_f1
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    # 创建一个handler
    handler = logging.StreamHandler()
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG，这样所有级别的日志都会被处理
    # 创建一个格式器，并添加到handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(handler)
    w, d, t, = 4, 64, 20
    # k = 10
    #europe
    # walk_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65,70,75,80,90,100,110,120,130,140,150]
    # europe
    # walk_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,70,80]
    #europe:
    # walk_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45,50]
    #usa:
    # walk_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] #zuihao
    # walk_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,55,60,65,70,75,80,85,90,95,100]
    # email-eu-core:
    # walk_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    #walk_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,55,60,65,70,75,80]
    walk_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,55,60,65,70,75,80,85,90,95,100]
    #walk_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,105,110,115,120]
    # walk_lengths = [4,8,12,16,20,24,28,32,36,40,44,48,52]
    network_name_dict = { '1': 'Europe', '2': 'usa', '3': 'email-eu-core','4':'football'}
    network_name_key = input(
        '选择使用的网络：1：Europe：，2：usa, 3:email-eu-core,4:football')
    if network_name_key not in network_name_dict:
        logger.error('请输入1或2或3或4')
        sys.exit(0)
    network_name = network_name_dict[network_name_key]
    if network_name == 'Europe':
        data = pd.read_csv('data/europe-airports.edgelist', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data)
        file_path = 'data/labels-europe-airports.txt'
    elif network_name == 'usa':
        data = pd.read_csv('data/usa-airports.edgelist',sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data)
        file_path = 'data/labels-usa-airports.txt'
    elif network_name == 'email-eu-core':
        data = pd.read_csv('data/email-Eu-core.txt',sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(str)
        data['target'] = data['target'].astype(str)
        G = nx.from_pandas_edgelist(data,create_using=nx.DiGraph())
        file_path = 'data/email-Eu-core-department-labels.txt'

    logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')

    # 分别对每一个数据集运行10次算法，并取10次结果的均值
    micro_f1_values = []
    macro_f1_values = []
    for _ in range(10):
        micro_f1 ,macro_f1=main()
        micro_f1_values.append(micro_f1)
        macro_f1_values.append(macro_f1)
    print(micro_f1_values)
    print(macro_f1_values)
    average_micro_f1 = sum(micro_f1_values)/len(micro_f1_values)
    average_macro_f1 = sum(macro_f1_values) / len(macro_f1_values)
    print("Average-Micro-F1 Score:", average_micro_f1)
    print("Average-Macro-F1 Score:", average_macro_f1)
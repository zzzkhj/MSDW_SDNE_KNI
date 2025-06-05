import logging
import random

import gensim
import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# def random_walk(G, v, l):
#     """
#     游走采样
#     G: 图    v: 游走的根节点    l: 游走的序列的长度
#     """
#     walk_seq = [v]
#     while len(walk_seq) < l:
#         node = walk_seq[-1]
#         neighbors = list(G.neighbors(node))
#         if len(neighbors) > 0:
#             candidates = []
#             for s in neighbors:
#                 if s not in walk_seq and G.degree(s) > 0:
#                     m = 1 / G.degree(s)
#                     if np.random.uniform(0, 1) < m:
#                         candidates.append(s)
#
#             if candidates:  # 如果存在候选节点
#                 # 随机选择一个候选节点
#                 next_node = random.choice(candidates)
#                 walk_seq.append(next_node)
#             else:
#                 break  # 如果没有候选节点，则跳出循环
#         else:
#             break  # 如果当前节点没有邻居节点，则跳出循环
#
#     return walk_seq


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


def get_edge_features(edges, node_embeddings):
    X = []
    for edge in edges:
        node1, node2 = edge
        print(f"Processing edge: {node1} - {node2}")
        print(f"Node1 in embeddings: {node1 in node_embeddings}")
        print(f"Node2 in embeddings: {node2 in node_embeddings}")

        # 检查节点是否存在于嵌入字典中
        if node1 in node_embeddings and node2 in node_embeddings:
            emb1 = node_embeddings[node1]
            emb2 = node_embeddings[node2]
            edge_features = np.concatenate((emb1, emb2))
            X.append(edge_features)
        else:
            print(f"Node {node1} or {node2} not found in embeddings.")
    return np.array(X)



if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    # 创建一个handler
    handler = logging.StreamHandler()
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG，这样所有级别的日志都会被处理
    # 创建一个格式器，并添加到handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(handler)
# 假设你已经有了一个名为'usa_graph'的NetworkX图
# 如果没有，你需要加载它或从其他来源获取它
    data = pd.read_csv('data/congress.txt', sep=' ', header=None)
    data.columns = ['source', 'target']
    data['source'] = data['source'].astype(pd.StringDtype())
    data['target'] = data['target'].astype(pd.StringDtype())
    G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
# 使用DeepWalk训练嵌入表示
# 定义窗口大小和迭代次数等参数
    w, d, t, = 4, 64, 10
    k = 10
    walk_lengths = [10,20,30]
    model = deep_walk(G, t, d, w)
    # 获取所有节点的嵌入表示
    node_embeddings = np.vstack([model.wv[str(node)] for node in G.nodes()])

    # 创建正样本（存在的边）
    positive_edges = list(G.edges())

    # 将positive_edges转换为二维numpy数组
    positive_edges_array = np.array(positive_edges)

    # 随机移除网络中50%的边作为测试样本
    test_edges_indices = np.random.choice(positive_edges_array.shape[0], size=int(positive_edges_array.shape[0] * 0.5),
                                          replace=False)
    test_edges = positive_edges_array[test_edges_indices]
    train_edges_indices = np.setdiff1d(np.arange(positive_edges_array.shape[0]), test_edges_indices)
    train_edges = positive_edges_array[train_edges_indices]

    # 构造负样本（不存在的边）
    negative_edges = []
    while len(negative_edges) < len(positive_edges):
        node1, node2 = np.random.choice(G.nodes(), size=2, replace=False)
        if (node1, node2) not in G.edges() and (node2, node1) not in G.edges():
            negative_edges.append((node1, node2))

            # 确保负样本是唯一的，并且没有与正样本重复
    negative_edges = list(set(negative_edges) - set(positive_edges))


    # 获取训练和测试集的特征
    X_train = get_edge_features(train_edges, node_embeddings)
    X_test = get_edge_features(test_edges, node_embeddings)

    # 标签编码：正样本为1，负样本为0
    y_train = [1] * len(train_edges) + [0] * len(negative_edges[:len(train_edges)])
    y_test = [1] * len(test_edges) + [0] * len(negative_edges[:len(test_edges)])

    # 对特征进行标准化处理
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练逻辑回归模型
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500)
    lr.fit(X_train_scaled, y_train)

    # 预测测试集
    y_pred = lr.predict_proba(X_test_scaled)[:, 1]

    # 计算AUC得分
    auc = roc_auc_score(y_test, y_pred)
    print(f'AUC Score: {auc:.4f}')



model = deep_walk(G, t, d, w)
import logging
import random
import sys
import gensim
import pandas as pd
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

#随机移除网络中50%的边作为测试样本，剩下的边作为训练样本。
def generate_positive_edges(graph, test_fraction=0.5):
    edges = list(graph.edges())
    np.random.shuffle(edges)
    test_size = int(len(edges) * test_fraction)
    return edges[test_size:], edges[:test_size]
#返回两个列表，第一个是训练集边，第二个是测试集边。

#通过构造与正样本相同数量且不存在于网络中的边作为负样本。
def generate_negative_edges(graph, num_edges):
    negative_edges = set()
    while len(negative_edges) < num_edges:
        i, j = np.random.choice(list(graph.nodes()), 2)
        if not graph.has_edge(i, j):
            negative_edges.add((i, j))
    return list(negative_edges)

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
                if s not in walk_seq and G.degree(s) > 0:
                    m = 1 / G.degree(s)
                    if np.random.uniform(0, 1) < m:
                        candidates.append(s)


            if not candidates:
                break

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
            #
            break

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


def main():
    # Generate positive and negative edges
    train_edges_pos, test_edges_pos = generate_positive_edges(G)
    num_neg_edges = len(train_edges_pos)
    train_edges_neg = generate_negative_edges(G, num_neg_edges)
    test_edges_neg = generate_negative_edges(G, len(test_edges_pos))

    # 将正样本边和负样本边分别合并
    train_edges = train_edges_pos + train_edges_neg
    test_edges = test_edges_pos + test_edges_neg
    #创建对应的标签，正样本标记为 1，负样本标记为 0
    labels_train = np.concatenate([np.ones(len(train_edges_pos)), np.zeros(len(train_edges_neg))])
    labels_test = np.concatenate([np.ones(len(test_edges_pos)), np.zeros(len(test_edges_neg))])

    # Generate node embeddings using DeepWalk
    embedding_model = deep_walk(G, t, d, w)

    # Get node embeddings
    embeddings = {str(node): embedding_model.wv[str(node)] for node in G.nodes()}

    # 构建训练集和测试集的特征向量集合
    X_train = np.array([np.concatenate([embeddings[str(edge[0])], embeddings[str(edge[1])]]) for edge in train_edges])
    X_test = np.array([np.concatenate([embeddings[str(edge[0])], embeddings[str(edge[1])]]) for edge in test_edges])

    #训练一个逻辑回归分类器，并使用L2正则化来防止过拟合，使用lbfgs求解器来进行参数优化，最多进行500次迭代来找到最佳的模型参数。
    model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=500)
    model.fit(X_train, labels_train)

    # 使用 model.predict_proba 方法预测测试集中每条边的存在概率。
    # 结果是一个二维数组，第一列是预测为负样本的概率，第二列是预测为正样本的概率。只要正样本的概率，因此提取第二列
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate AUC score 计算测试集中真实标签 labels_test 和预测概率 y_pred_proba 之间的 AUC
    auc_score = roc_auc_score(labels_test, y_pred_proba)

    return  auc_score
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
    w, d, t = 4, 64, 50
    # walk_lengths = [3, 6, 10, 13, 17, 20, 24] #karate club
    # walk_lengths = [3,6,10,14,18,22,26,30,34] #dolphins
    # walk_lengths =  [4,8,12,16,20,24,28,32,36,40,44,48,52] #jazz
    # walk_lengths =  [5,10,15,20,25, 30,35, 40,45,50, 60, 70, 80, 90, 100,110,120] #europe
    # walk_lengths =  [5,10,15,20,25, 30,35, 40,45,50, 60, 70, 80, 90, 100,110,120,130,140,150] #usa
    # walk_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,160,170,180,190,200]  # usa  and  facebook
    walk_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,150]
    network_name_dict = {'1': 'Europe', '2': 'usa', '3': 'karate club', '4': 'dolphins', '5': 'jazz', '6': 'facebook','7':'congress','8':'email-eu-core'}
    network_name_key = input(
        '选择使用的网络：1：Europe，2：usa, 3:karate club,4:dolphins,5:jazz,6:facebook,7:congress,8:email-eu-core')
    if network_name_key not in network_name_dict:
        logger.error('请输入网络编号')
        sys.exit(0)
    network_name = network_name_dict[network_name_key]
    if network_name == 'Europe':
        data = pd.read_csv('data/europe-airports.edgelist', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data)

    elif network_name == 'usa':
        data = pd.read_csv('data/usa-airports.edgelist', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(pd.StringDtype())
        data['target'] = data['target'].astype(pd.StringDtype())
        G = nx.from_pandas_edgelist(data)

    elif network_name == 'karate club':
        data = pd.read_csv('data/karate.txt', sep='\t', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(str)
        data['target'] = data['target'].astype(str)
        G = nx.from_pandas_edgelist(data)
    elif network_name == 'dolphins':
        data = pd.read_csv('data/dolphins.csv', sep=',', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(str)
        data['target'] = data['target'].astype(str)
        G = nx.from_pandas_edgelist(data)
    elif network_name == 'jazz':
        data = pd.read_csv('data/jazz.txt', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(str)
        data['target'] = data['target'].astype(str)
        G = nx.from_pandas_edgelist(data)
    elif network_name == 'congress':
        data = pd.read_csv('data/congress.txt', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(str)
        data['target'] = data['target'].astype(str)
        G = nx.from_pandas_edgelist(data)
    elif network_name == 'facebook':
        data = pd.read_csv('data/facebook_combined.txt', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(str)
        data['target'] = data['target'].astype(str)
        G = nx.from_pandas_edgelist(data)
    elif network_name == 'email-eu-core':
        data = pd.read_csv('data/email-Eu-core.txt', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(str)
        data['target'] = data['target'].astype(str)
        G = nx.from_pandas_edgelist(data, create_using=nx.DiGraph())
    logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')

    total_auc_score = 0  # 初始化AUC总分数的变量
    #分别对每一个数据集运行10次算法，并取10次结果的均值
    for _ in range(10):
        auc_score = main()  # main()函数返回一个AUC分数
        if auc_score is not None:
            total_auc_score += auc_score  # 累加AUC分数
        else:
            print("Warning: AUC score is None. Skipping this iteration.")
    print(total_auc_score)
    average_auc_score = total_auc_score / 10  # 计算平均AUC分数
    print("Average AUC score:", average_auc_score)
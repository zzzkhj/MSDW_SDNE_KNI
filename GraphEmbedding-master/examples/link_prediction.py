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
from ge.classify import read_node_label, Classifier
from ge import SDNE,Struc2Vec
from sklearn.manifold import TSNE
def generate_positive_edges(graph, test_fraction=0.5):
    edges = list(graph.edges())
    np.random.shuffle(edges)
    test_size = int(len(edges) * test_fraction)
    return edges[test_size:], edges[:test_size]

def generate_negative_edges(graph, num_edges):
    negative_edges = set()
    while len(negative_edges) < num_edges:
        i, j = np.random.choice(list(graph.nodes()), 2)
        if not graph.has_edge(i, j):
            negative_edges.add((i, j))
    return list(negative_edges)


def main():
    # Load data and construct graph
    # data = load_data('data/karate.txt')
    # G = nx.from_pandas_edgelist(data, 'source', 'target')

    # Generate positive and negative edges
    train_edges_pos, test_edges_pos = generate_positive_edges(G)
    num_neg_edges = len(train_edges_pos)
    train_edges_neg = generate_negative_edges(G, num_neg_edges)
    test_edges_neg = generate_negative_edges(G, len(test_edges_pos))

    # Combine positive and negative edges and create labels
    train_edges = train_edges_pos + train_edges_neg
    test_edges = test_edges_pos + test_edges_neg
    labels_train = np.concatenate([np.ones(len(train_edges_pos)), np.zeros(len(train_edges_neg))])
    labels_test = np.concatenate([np.ones(len(test_edges_pos)), np.zeros(len(test_edges_neg))])

    # Generate node embeddings using DeepWalk
    # embedding_model = SDNE(G, hidden_size=[256, 64], )
    #embedding_model.train(batch_size=3000, epochs=100, verbose=2)
    embedding_model = Struc2Vec(G)
    embedding_model.train()
    embeddings = embedding_model.get_embeddings()

    # Prepare feature vectors for training and testing
    X_train = np.array([np.concatenate([embeddings[str(edge[0])], embeddings[str(edge[1])]]) for edge in train_edges])
    X_test = np.array([np.concatenate([embeddings[str(edge[0])], embeddings[str(edge[1])]]) for edge in test_edges])

    # Train logistic regression model
    model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=500)
    model.fit(X_train, labels_train)

    # Predict probabilities for the test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate AUC score
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

    network_name_dict = {'1': 'Europe', '2': 'usa', '3': 'karate club', '4': 'dolphins', '5': 'jazz', '6': 'facebook','7':'congress'}
    network_name_key = input(
        '选择使用的网络：1：Europe，2：usa, 3:karate club,4:dolphins,5:jazz,6:facebook,7:congress')
    if network_name_key not in network_name_dict:
        logger.error('请输入网络编号')
        sys.exit(0)
    network_name = network_name_dict[network_name_key]
    if network_name == 'Europe':
        data = pd.read_csv('data/europe-airports.edgelist', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(str)
        data['target'] = data['target'].astype(str)
        G = nx.from_pandas_edgelist(data)

    elif network_name == 'usa':
        data = pd.read_csv('data/usa-airports.edgelist', sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(str)
        data['target'] = data['target'].astype(str)
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
    logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')

    total_auc_score = 0  # 初始化AUC总分数的变量

    for _ in range(10):
        auc_score = main()  # 假设main()返回一个AUC分数
        if auc_score is not None:  # 检查auc_score不是None
            total_auc_score += auc_score  # 累加AUC分数
        else:
            print("Warning: AUC score is None. Skipping this iteration.")
    print(total_auc_score)
    average_auc_score = total_auc_score / 10  # 计算平均AUC分数
    print("Average AUC score:", average_auc_score)
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
from ge.classify import read_node_label, Classifier
from ge import SDNE
from sklearn.manifold import TSNE
import numpy as np
from ge import Struc2Vec

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

def main():
    # Generate node embeddings   # Get node embeddings
    # embedding_model = SDNE(G, hidden_size=[256, 64], )
    # embedding_model.train(batch_size=3000, epochs=40, verbose=2)
    # embeddings = embedding_model.get_embeddings()

    model = Struc2Vec(G, 10, 80, workers=4, verbose=40, )
    model.train()
    embeddings = model.get_embeddings()
    # Generate node labels
    # node_labels = read_node_labels('data/labels-usa-airports.txt')file_path
    node_labels = read_node_labels(file_path)

    # Prepare feature vectors and labels
    X = np.array([embeddings[str(node)] for node in G.nodes()])
    y = np.array([node_labels[str(node)] for node in G.nodes()])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5,random_state=42)

    # Train One-Vs-Rest logistic regression model
    model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=10000, multi_class='ovr')
    model.fit(X_train, y_train)

    # Predict labels for test set
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
        data['source'] = data['source'].astype(str)
        data['target'] = data['target'].astype(str)
        G = nx.from_pandas_edgelist(data)
        file_path = 'data/labels-europe-airports.txt'
    elif network_name == 'usa':
        data = pd.read_csv('data/usa-airports.edgelist',sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(str)
        data['target'] = data['target'].astype(str)
        G = nx.from_pandas_edgelist(data)
        file_path = 'data/labels-usa-airports.txt'
    elif network_name == 'email-eu-core':
        data = pd.read_csv('data/email-Eu-core.txt',sep=' ', header=None)
        data.columns = ['source', 'target']
        data['source'] = data['source'].astype(str)
        data['target'] = data['target'].astype(str)
        G = nx.from_pandas_edgelist(data,create_using=nx.DiGraph())
        file_path = 'data/email-Eu-core-department-labels.txt'
    # elif network_name == 'football':
    #     G = nx.read_gml('data/football.gml')
    #     # file_path = 'data/football.gml'
    #     labels = nx.get_node_attributes(G, 'label')
    logger.info(f'使用{network_name}数据集，图有{G.number_of_nodes()}个节点，{G.number_of_edges()}条边。')



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